// SPDX-License-Identifier: Apache-2.0

#include <infiniband/verbs.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#if defined(ENABLE_ASCEND_ACL)
#include <acl/acl.h>

#ifndef ACL_HOST_REG_MAPPED
#define ACL_HOST_REG_MAPPED 0x2UL
#endif

#ifndef ACL_HOST_REG_PINNED
#define ACL_HOST_REG_PINNED 0x10000000UL
#endif
#endif

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

namespace {

struct Options {
  std::string device;
  uint8_t port = 1;
  size_t size = 8UL * 1024 * 1024;
  std::string allocator = "aligned";
  bool remote_access = true;
  int32_t acl_device = 0;
  bool acl_register_v2 = false;
};

class AclRuntime {
 public:
  AclRuntime(bool required, int32_t device_id) {
    if (!required) {
      return;
    }
#if defined(ENABLE_ASCEND_ACL)
    const aclError error = aclInit(nullptr);
    if (error != ACL_SUCCESS) {
      throw std::runtime_error("aclInit failed: error=" +
                               std::to_string(error));
    }
    initialized_ = true;
    device_id_ = device_id;

    const aclError set_device_error = aclrtSetDevice(device_id_);
    if (set_device_error != ACL_SUCCESS) {
      aclFinalize();
      initialized_ = false;
      throw std::runtime_error("aclrtSetDevice failed: device=" +
                               std::to_string(device_id_) + " error=" +
                               std::to_string(set_device_error));
    }
    device_set_ = true;
#else
    static_cast<void>(device_id);
    throw std::runtime_error(
        "ACL operations require compiling with -DENABLE_ASCEND_ACL");
#endif
  }

  AclRuntime(const AclRuntime&) = delete;
  AclRuntime& operator=(const AclRuntime&) = delete;

  ~AclRuntime() {
#if defined(ENABLE_ASCEND_ACL)
    if (device_set_) {
      const aclError error = aclrtResetDevice(device_id_);
      if (error != ACL_SUCCESS) {
        std::cerr << "aclrtResetDevice FAILED: device=" << device_id_
                  << " error=" << error << '\n';
      }
    }
    if (initialized_) {
      const aclError error = aclFinalize();
      if (error != ACL_SUCCESS) {
        std::cerr << "aclFinalize FAILED: error=" << error << '\n';
      }
    }
#endif
  }

 private:
#if defined(ENABLE_ASCEND_ACL)
  bool initialized_ = false;
  bool device_set_ = false;
  int32_t device_id_ = 0;
#endif
};

class AclHostRegistration {
 public:
  AclHostRegistration(bool enabled, void* address, size_t size) {
    if (!enabled) {
      return;
    }
#if defined(ENABLE_ASCEND_ACL)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    }
    const auto page_size_bytes = static_cast<uintptr_t>(page_size);
    const auto address_value = reinterpret_cast<uintptr_t>(address);
    if (address_value % page_size_bytes != 0) {
      throw std::invalid_argument(
          "aclrtHostRegisterV2 requires a page-aligned address");
    }
    if (size % static_cast<size_t>(page_size) != 0) {
      throw std::invalid_argument(
          "aclrtHostRegisterV2 requires a page-multiple size");
    }

    constexpr uint32_t flags = ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED;
    const aclError error = aclrtHostRegisterV2(address, size, flags);
    if (error != ACL_SUCCESS) {
      throw std::runtime_error("aclrtHostRegisterV2 failed: error=" +
                               std::to_string(error));
    }
    address_ = address;
    registered_ = true;
    std::cout << "aclrtHostRegisterV2 SUCCEEDED: flags=0x" << std::hex
              << flags << std::dec << '\n';
#else
    static_cast<void>(address);
    static_cast<void>(size);
    throw std::runtime_error(
        "--acl-register-v2 requires compiling with -DENABLE_ASCEND_ACL");
#endif
  }

  AclHostRegistration(const AclHostRegistration&) = delete;
  AclHostRegistration& operator=(const AclHostRegistration&) = delete;

  ~AclHostRegistration() { unregister(); }

  bool unregister() {
#if defined(ENABLE_ASCEND_ACL)
    if (!registered_) {
      return true;
    }
    const aclError error = aclrtHostUnregister(address_);
    if (error != ACL_SUCCESS) {
      std::cerr << "aclrtHostUnregister FAILED: error=" << error << '\n';
      return false;
    }
    registered_ = false;
    std::cout << "aclrtHostUnregister SUCCEEDED\n";
#endif
    return true;
  }

 private:
#if defined(ENABLE_ASCEND_ACL)
  void* address_ = nullptr;
  bool registered_ = false;
#endif
};

class Buffer {
 public:
  Buffer(size_t size, const std::string& allocator)
      : size_(size), allocator_(allocator) {
    if (allocator == "aligned") {
      const long page_size = sysconf(_SC_PAGESIZE);
      if (page_size <= 0) {
        throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
      }
      const int rc = posix_memalign(&address_, static_cast<size_t>(page_size),
                                    size_);
      if (rc != 0) {
        throw std::runtime_error("posix_memalign failed: " +
                                 std::string(std::strerror(rc)));
      }
    } else if (allocator == "mmap") {
      address_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (address_ == MAP_FAILED) {
        address_ = nullptr;
        throw_system_error("anonymous mmap");
      }
    } else if (allocator == "huge2m") {
      address_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
                      -1, 0);
      if (address_ == MAP_FAILED) {
        address_ = nullptr;
        throw_system_error("2 MiB hugepage mmap");
      }
    } else if (allocator == "acl") {
#if defined(ENABLE_ASCEND_ACL)
      const aclError error = aclrtMallocHost(&address_, size_);
      if (error != ACL_SUCCESS || address_ == nullptr) {
        address_ = nullptr;
        throw std::runtime_error("aclrtMallocHost failed: error=" +
                                 std::to_string(error));
      }
#else
      throw std::runtime_error(
          "allocator 'acl' requires compiling with -DENABLE_ASCEND_ACL");
#endif
    } else {
      throw std::invalid_argument("unknown allocator: " + allocator);
    }

    // Fault in every base page before registration. This makes allocation
    // failures visible before ibv_reg_mr() and resembles a buffer in real use.
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    }
    auto* bytes = static_cast<volatile unsigned char*>(address_);
    for (size_t offset = 0; offset < size_;
         offset += static_cast<size_t>(page_size)) {
      bytes[offset] = 0;
    }
    bytes[size_ - 1] = 0;
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer() {
    if (address_ == nullptr) {
      return;
    }
    if (allocator_ == "aligned") {
      free(address_);
    } else if (allocator_ == "acl") {
#if defined(ENABLE_ASCEND_ACL)
      const aclError error = aclrtFreeHost(address_);
      if (error != ACL_SUCCESS) {
        std::cerr << "aclrtFreeHost FAILED: error=" << error << '\n';
      }
#endif
    } else {
      munmap(address_, size_);
    }
  }

  void* address() const { return address_; }
  size_t size() const { return size_; }

 private:
  [[noreturn]] static void throw_system_error(const std::string& operation) {
    const int error = errno;
    throw std::runtime_error(operation + " failed: errno=" +
                             std::to_string(error) + " (" +
                             std::string(std::strerror(error)) + ")");
  }

  void* address_ = nullptr;
  size_t size_;
  std::string allocator_;
};

[[noreturn]] void usage(const char* program, int exit_code) {
  std::ostream& output = exit_code == EXIT_SUCCESS ? std::cout : std::cerr;
  output << "Usage: " << program << " --device <name> [options]\n"
         << "\n"
         << "Register one local memory region with libibverbs. No remote peer "
            "is needed.\n"
         << "\n"
         << "Options:\n"
         << "  -d, --device NAME       RDMA device, for example mlx5_0 "
            "(required)\n"
         << "  -p, --port NUMBER       RDMA port to validate (default: 1)\n"
         << "  -s, --size SIZE         Bytes or K/M/G suffix (default: 8M)\n"
         << "  -a, --allocator TYPE    aligned, mmap, huge2m, or acl "
            "(default: aligned)\n"
         << "      --acl-device ID     Ascend device for ACL setup (default: 0)\n"
         << "      --acl-register-v2   Register externally allocated memory "
            "with ACL V2\n"
         << "      --local-only        Register only IBV_ACCESS_LOCAL_WRITE\n"
         << "  -h, --help              Show this help\n"
         << "\n"
         << "The default access flags match Mooncake RDMA registration:\n"
         << "IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | "
            "IBV_ACCESS_REMOTE_WRITE.\n";
  std::exit(exit_code);
}

uint64_t parse_unsigned(const std::string& text, const char* option_name) {
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(std::string(option_name) +
                                " must be a positive integer");
  }
  size_t parsed = 0;
  const unsigned long long value = std::stoull(text, &parsed, 10);
  if (parsed != text.size()) {
    throw std::invalid_argument(std::string("invalid ") + option_name +
                                ": " + text);
  }
  return value;
}

size_t parse_size(const std::string& text) {
  if (text.empty()) {
    throw std::invalid_argument("size must not be empty");
  }

  uint64_t multiplier = 1;
  std::string number = text;
  const char suffix = text.back();
  if (suffix == 'K' || suffix == 'k') {
    multiplier = 1024;
    number.pop_back();
  } else if (suffix == 'M' || suffix == 'm') {
    multiplier = 1024 * 1024;
    number.pop_back();
  } else if (suffix == 'G' || suffix == 'g') {
    multiplier = 1024ULL * 1024 * 1024;
    number.pop_back();
  }

  const uint64_t value = parse_unsigned(number, "size");
  if (value == 0 || value > std::numeric_limits<size_t>::max() / multiplier) {
    throw std::invalid_argument("size is zero or too large: " + text);
  }
  return static_cast<size_t>(value * multiplier);
}

Options parse_options(int argc, char** argv) {
  Options options;
  const option long_options[] = {
      {"device", required_argument, nullptr, 'd'},
      {"port", required_argument, nullptr, 'p'},
      {"size", required_argument, nullptr, 's'},
      {"allocator", required_argument, nullptr, 'a'},
      {"acl-device", required_argument, nullptr, 1001},
      {"acl-register-v2", no_argument, nullptr, 1002},
      {"local-only", no_argument, nullptr, 1000},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  while (true) {
    const int value = getopt_long(argc, argv, "d:p:s:a:h", long_options, nullptr);
    if (value == -1) {
      break;
    }
    switch (value) {
      case 'd':
        options.device = optarg;
        break;
      case 'p': {
        const uint64_t port = parse_unsigned(optarg, "port");
        if (port == 0 || port > std::numeric_limits<uint8_t>::max()) {
          throw std::invalid_argument("port must be between 1 and 255");
        }
        options.port = static_cast<uint8_t>(port);
        break;
      }
      case 's':
        options.size = parse_size(optarg);
        break;
      case 'a':
        options.allocator = optarg;
        break;
      case 1000:
        options.remote_access = false;
        break;
      case 1001: {
        const uint64_t device_id = parse_unsigned(optarg, "acl-device");
        if (device_id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          throw std::invalid_argument("acl-device is too large");
        }
        options.acl_device = static_cast<int32_t>(device_id);
        break;
      }
      case 1002:
        options.acl_register_v2 = true;
        break;
      case 'h':
        usage(argv[0], EXIT_SUCCESS);
      default:
        usage(argv[0], EXIT_FAILURE);
    }
  }

  if (options.device.empty()) {
    throw std::invalid_argument("--device is required");
  }
  if (options.acl_register_v2 && options.allocator == "acl") {
    throw std::invalid_argument(
        "--acl-register-v2 requires externally allocated memory; use "
        "aligned, mmap, or huge2m instead of acl");
  }
  if (optind != argc) {
    throw std::invalid_argument("unexpected positional argument: " +
                                std::string(argv[optind]));
  }
  return options;
}

const char* port_state_name(ibv_port_state state) {
  switch (state) {
    case IBV_PORT_DOWN:
      return "DOWN";
    case IBV_PORT_INIT:
      return "INIT";
    case IBV_PORT_ARMED:
      return "ARMED";
    case IBV_PORT_ACTIVE:
      return "ACTIVE";
    case IBV_PORT_ACTIVE_DEFER:
      return "ACTIVE_DEFER";
    default:
      return "UNKNOWN";
  }
}

ibv_device* find_device(ibv_device** devices, int count,
                        const std::string& requested) {
  std::cerr << "Discovered RDMA devices:";
  for (int i = 0; i < count; ++i) {
    const char* name = ibv_get_device_name(devices[i]);
    std::cerr << (i == 0 ? " " : ", ") << name;
    if (requested == name) {
      std::cerr << '\n';
      return devices[i];
    }
  }
  std::cerr << '\n';
  throw std::runtime_error("RDMA device not found: " + requested);
}

int run(const Options& options) {
  int device_count = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr) {
    const int error = errno;
    throw std::runtime_error("ibv_get_device_list failed: errno=" +
                             std::to_string(error) + " (" +
                             std::string(std::strerror(error)) + ")");
  }

  ibv_device* device = nullptr;
  try {
    device = find_device(devices, device_count, options.device);
  } catch (...) {
    ibv_free_device_list(devices);
    throw;
  }

  ibv_context* context = ibv_open_device(device);
  ibv_free_device_list(devices);
  if (context == nullptr) {
    const int error = errno;
    throw std::runtime_error("ibv_open_device failed: errno=" +
                             std::to_string(error) + " (" +
                             std::string(std::strerror(error)) + ")");
  }

  ibv_device_attr device_attr{};
  if (ibv_query_device(context, &device_attr) != 0) {
    const int error = errno;
    ibv_close_device(context);
    throw std::runtime_error("ibv_query_device failed: errno=" +
                             std::to_string(error) + " (" +
                             std::string(std::strerror(error)) + ")");
  }

  ibv_port_attr port_attr{};
  if (ibv_query_port(context, options.port, &port_attr) != 0) {
    const int error = errno;
    ibv_close_device(context);
    throw std::runtime_error("ibv_query_port failed: errno=" +
                             std::to_string(error) + " (" +
                             std::string(std::strerror(error)) + ")");
  }

  std::cout << "device=" << options.device
            << " port=" << static_cast<unsigned int>(options.port)
            << " state=" << port_state_name(port_attr.state)
            << " max_mr_size=" << device_attr.max_mr_size
            << " max_mr=" << device_attr.max_mr << '\n';
  if (port_attr.state != IBV_PORT_ACTIVE) {
    std::cerr << "WARNING: selected port is not ACTIVE. Local MR registration "
                 "may still work, but this port cannot transfer data.\n";
  }

  ibv_pd* protection_domain = ibv_alloc_pd(context);
  if (protection_domain == nullptr) {
    const int error = errno;
    ibv_close_device(context);
    throw std::runtime_error("ibv_alloc_pd failed: errno=" +
                             std::to_string(error) + " (" +
                             std::string(std::strerror(error)) + ")");
  }

  int result = EXIT_FAILURE;
  try {
    // Objects are destroyed in reverse order: ACL registration, buffer, then
    // ACL runtime. RDMA registration is explicitly removed first below.
    const bool acl_required =
        options.allocator == "acl" || options.acl_register_v2;
    AclRuntime acl_runtime(acl_required, options.acl_device);
    Buffer buffer(options.size, options.allocator);
    AclHostRegistration acl_registration(
        options.acl_register_v2, buffer.address(), buffer.size());
    const auto address = reinterpret_cast<uintptr_t>(buffer.address());
    int access = IBV_ACCESS_LOCAL_WRITE;
    if (options.remote_access) {
      access |= IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    }

    std::cout << "allocator=" << options.allocator << " address=0x" << std::hex
              << address << std::dec << " size=" << buffer.size()
              << " page_offset="
              << address % static_cast<uintptr_t>(sysconf(_SC_PAGESIZE))
              << " access=0x" << std::hex << access << std::dec << '\n';

    errno = 0;
    ibv_mr* memory_region =
        ibv_reg_mr(protection_domain, buffer.address(), buffer.size(), access);
    const int registration_error = errno;
    if (memory_region == nullptr) {
      std::cerr << "ibv_reg_mr FAILED: errno=" << registration_error << " ("
                << std::strerror(registration_error) << ")\n";
    } else {
      std::cout << "ibv_reg_mr SUCCEEDED: lkey=0x" << std::hex
                << memory_region->lkey << " rkey=0x" << memory_region->rkey
                << std::dec << '\n';
      if (ibv_dereg_mr(memory_region) != 0) {
        const int error = errno;
        std::cerr << "ibv_dereg_mr FAILED: errno=" << error << " ("
                  << std::strerror(error) << ")\n";
      } else {
        std::cout << "ibv_dereg_mr SUCCEEDED\n";
        result = acl_registration.unregister() ? EXIT_SUCCESS : EXIT_FAILURE;
      }
    }
  } catch (...) {
    ibv_dealloc_pd(protection_domain);
    ibv_close_device(context);
    throw;
  }

  if (ibv_dealloc_pd(protection_domain) != 0) {
    const int error = errno;
    std::cerr << "ibv_dealloc_pd FAILED: errno=" << error << " ("
              << std::strerror(error) << ")\n";
    result = EXIT_FAILURE;
  }
  if (ibv_close_device(context) != 0) {
    const int error = errno;
    std::cerr << "ibv_close_device FAILED: errno=" << error << " ("
              << std::strerror(error) << ")\n";
    result = EXIT_FAILURE;
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
