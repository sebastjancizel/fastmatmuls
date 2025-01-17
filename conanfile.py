from conan import ConanFile


class MatmulConan(ConanFile):
    name = "matmul"
    version = "0.1.0"
    license = "MIT"
    author = "Sebastjan Cizel <sebastjan.cizel@gmail.com>"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("gtest/1.15.0")
        self.requires("benchmark/1.9.0")

    def build_requirements(self):
        self.build_requires("cmake/3.30.5")

    def layout(self):
        self.folders.generators = "generators"
