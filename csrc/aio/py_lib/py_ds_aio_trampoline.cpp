#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "deepspeed_aio_base.h"
#include <string>
#include <iostream>
#include <memory>
#include <dlfcn.h>
#include <filesystem>

namespace py = pybind11;

class DeepSpeedAIOTrampoline : public DeepSpeedAIOBase {
public:
    DeepSpeedAIOTrampoline(const std::string& device_type) : device(nullptr), handle(nullptr) {
        load_device(device_type);
    }

    void load_device(const std::string& device_type) {
        if (device) {
            delete device;
        }

        if (handle) {
            dlclose(handle);
        }

        std::filesystem::path so_directory = std::filesystem::current_path() / "deepspeed" / "ops" / "plugins";

        std::filesystem::path lib_path = so_directory / (device_type + "_op.so");

        handle = dlopen(lib_name.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Cannot open library: " << dlerror() << '\n';
            return;
        }

        dlerror();

        typedef DeepSpeedAIOBase* (*create_t)();

        create_t create_device = (create_t) dlsym(handle, "create_device");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            std::cerr << "Cannot load symbol create_device: " << dlsym_error << '\n';
            dlclose(handle);
            handle = nullptr;
            return;
        }

        device = create_device();
    }

    void aio_read(torch::Tensor& buffer, const char* filename, const bool validate) override { device->aio_read(buffer, filename, validate); }
    void aio_write(const torch::Tensor& buffer, const char* filename, const bool validate) override { device->aio_write(buffer, filename, validate); }
    void deepspeed_memcpy(torch::Tensor& dest, const torch::Tensor& src) override { device->deepspeed_memcpy(dest, src); }
    int get_block_size() const override { return device->get_block_size(); }
    int get_queue_depth() const override { return device->get_queue_depth(); }
    bool get_single_submit() const override { return device->get_single_submit(); }
    bool get_overlap_events() const override { return device->get_overlap_events(); }
    int get_thread_count() const override { return device->get_thread_count(); }
    void read(torch::Tensor& buffer, const char* filename, const bool validate) override { device->read(buffer, filename, validate); }
    void write(const torch::Tensor& buffer, const char* filename, const bool validate) override { device->write(buffer, filename, validate); }
    void pread(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) override { device->pread(buffer, filename, validate, async); }
    void pwrite(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) override { device->pwrite(buffer, filename, validate, async); }
    void sync_pread(torch::Tensor& buffer, const char* filename) override { device->sync_pread(buffer, filename); }
    void sync_pwrite(const torch::Tensor& buffer, const char* filename) override { device->sync_pwrite(buffer, filename); }
    void async_pread(torch::Tensor& buffer, const char* filename) override { device->async_pread(buffer, filename); }
    void async_pwrite(const torch::Tensor& buffer, const char* filename) override { device->async_pwrite(buffer, filename); }
    void new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor) override { device->new_cpu_locked_tensor(num_elem, example_tensor); }
    void free_cpu_locked_tensor(torch::Tensor& tensor) override { device->free_cpu_locked_tensor(tensor); }
    void wait() override { device->wait(); }

    ~DeepSpeedAIOTrampoline() {
        if (device) {
            delete device;
        }
        if (handle) {
            dlclose(handle);
        }
    }

private:
    DeepSpeedAIOBase* device;
    void* handle;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<DeepSpeedAIOTrampoline>(m, "DeepSpeedAIOTrampoline")
        .def(py::init<const std::string&>(), py::arg("device_type") = "nvme")
        .def(py::init<const std::string&>())
        .def("load_device", &DeepSpeedAIOTrampoline::load_device)
        .def("aio_read", &DeepSpeedAIOTrampoline::aio_read, py::arg("buffer"), py::arg("filename"), py::arg("validate") = true)
        .def("aio_write", &DeepSpeedAIOTrampoline::aio_write, py::arg("buffer"), py::arg("filename"), py::arg("validate") = true)
        .def("deepspeed_memcpy", &DeepSpeedAIOTrampoline::deepspeed_memcpy)
        .def("get_block_size", &DeepSpeedAIOTrampoline::get_block_size)
        .def("get_queue_depth", &DeepSpeedAIOTrampoline::get_queue_depth)
        .def("get_single_submit", &DeepSpeedAIOTrampoline::get_single_submit)
        .def("get_overlap_events", &DeepSpeedAIOTrampoline::get_overlap_events)
        .def("get_thread_count", &DeepSpeedAIOTrampoline::get_thread_count)
        .def("read", &DeepSpeedAIOTrampoline::read)
        .def("write", &DeepSpeedAIOTrampoline::write)
        .def("pread", &DeepSpeedAIOTrampoline::pread, py::arg("buffer"), py::arg("filename"), py::arg("validate") = true, py::arg("async") = false)
        .def("pwrite", &DeepSpeedAIOTrampoline::pwrite, py::arg("buffer"), py::arg("filename"), py::arg("validate") = true, py::arg("async") = false)
        .def("sync_pread", &DeepSpeedAIOTrampoline::sync_pread, py::arg("buffer"), py::arg("filename"))
        .def("sync_pwrite", &DeepSpeedAIOTrampoline::sync_pwrite, py::arg("buffer"), py::arg("filename"))
        .def("async_pread", &DeepSpeedAIOTrampoline::async_pread, py::arg("buffer"), py::arg("filename"))
        .def("async_pwrite", &DeepSpeedAIOTrampoline::async_pwrite, py::arg("buffer"), py::arg("filename"))
        .def("new_cpu_locked_tensor", &DeepSpeedAIOTrampoline::new_cpu_locked_tensor, py::arg("num_elem"), py::arg("example_tensor"))
        .def("free_cpu_locked_tensor", &DeepSpeedAIOTrampoline::free_cpu_locked_tensor, py::arg("tensor"))
        .def("wait", &DeepSpeedAIOTrampoline::wait);
}







// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include "deepspeed_aio_base.h"
// #include <string>
// #include <iostream>
// #include <memory>
// #include <dlfcn.h>
// #include<filesystem>

// namespace py = pybind11;

// class DeepSpeedAIOTrampoline : public DeepSpeedAIOBase {
// public:
//     DeepSpeedAIOTrampoline(const std::string& device_type) : device(nullptr), handle(nullptr) {
//         load_device(device_type);
//     }

//     void load_device(const std::string& device_type) {
//         if (device) {
//             delete device;
//         }

//         if (handle) {
//             dlclose(handle);
//         }

//         std::filesystem::path so_directory = std::filesystem::current_path() / "deepspeed" / "ops" / "plugins";

//         std::filesystem::path lib_path = so_directory / (device_type + "_op.so");

//         handle = dlopen(lib_name.c_str(), RTLD_LAZY);
//         if (!handle) {
//             std::cerr << "Cannot open library: " << dlerror() << '\n';
//             return;
//         }

//         dlerror();

//         typedef DeepSpeedAIOBase* (*create_t)();

//         create_t create_device = (create_t) dlsym(handle, "create_device");
//         const char* dlsym_error = dlerror();
//         if (dlsym_error) {
//             std::cerr << "Cannot load symbol create_device: " << dlsym_error << '\n';
//             dlclose(handle);
//             handle = nullptr;
//             return;
//         }

//         device = create_device();
//     }

//     void aio_read() override { device->aio_read(); }
//     void aio_write() override { device->aio_write(); }
//     void deepspeed_memcpy() override { device->deepspeed_memcpy(); }
//     int get_block_size() const override { return device->get_block_size(); }
//     int get_queue_depth() const override { return device->get_queue_depth(); }
//     bool get_single_submit() const override { return device->get_single_submit(); }
//     bool get_overlap_events() const override { return device->get_overlap_events(); }
//     int get_thread_count() const override { return device->get_thread_count(); }
//     void read() override { device->read(); }
//     void write() override { device->write(); }
//     void pread() override { device->pread(); }
//     void pwrite() override { device->pwrite(); }
//     void sync_pread() override { device->sync_pread(); }
//     void sync_pwrite() override { device->sync_pwrite(); }
//     void async_pread() override { device->async_pread(); }
//     void async_pwrite() override { device->async_pwrite(); }
//     void new_cpu_locked_tensor() override { device->new_cpu_locked_tensor(); }
//     void free_cpu_locked_tensor() override { device->free_cpu_locked_tensor(); }
//     void wait() override { device->wait(); }

//     ~DeepSpeedAIOTrampoline() {
//         if (device) {
//             delete device;
//         }
//         if (handle) {
//             dlclose(handle);
//         }
//     }

// private:
//     DeepSpeedAIOBase* device;
//     void* handle;
// };

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     py::class_<DeepSpeedAIOTrampoline>(m, "DeepSpeedAIOTrampoline")
//         .def(py::init<const std::string&>(), py::arg("device_type") = "nvme")
//         .def(py::init<const std::string&>())
//         .def("load_device", &DeepSpeedAIOTrampoline::load_device)
//         .def("aio_read", &DeepSpeedAIOTrampoline::aio_read)
//         .def("aio_write", &DeepSpeedAIOTrampoline::aio_write)
//         .def("deepspeed_memcpy", &DeepSpeedAIOTrampoline::deepspeed_memcpy)
//         .def("get_block_size", &DeepSpeedAIOTrampoline::get_block_size)
//         .def("get_queue_depth", &DeepSpeedAIOTrampoline::get_queue_depth)
//         .def("get_single_submit", &DeepSpeedAIOTrampoline::get_single_submit)
//         .def("get_overlap_events", &DeepSpeedAIOTrampoline::get_overlap_events)
//         .def("get_thread_count", &DeepSpeedAIOTrampoline::get_thread_count)
//         .def("read", &DeepSpeedAIOTrampoline::read)
//         .def("write", &DeepSpeedAIOTrampoline::write)
//         .def("pread", &DeepSpeedAIOTrampoline::pread)
//         .def("pwrite", &DeepSpeedAIOTrampoline::pwrite)
//         .def("sync_pread", &DeepSpeedAIOTrampoline::sync_pread)
//         .def("sync_pwrite", &DeepSpeedAIOTrampoline::sync_pwrite)
//         .def("async_pread", &DeepSpeedAIOTrampoline::async_pread)
//         .def("async_pwrite", &DeepSpeedAIOTrampoline::async_pwrite)
//         .def("new_cpu_locked_tensor", &DeepSpeedAIOTrampoline::new_cpu_locked_tensor)
//         .def("free_cpu_locked_tensor", &DeepSpeedAIOTrampoline::free_cpu_locked_tensor)
//         .def("wait", &DeepSpeedAIOTrampoline::wait);
// }
