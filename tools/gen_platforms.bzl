"""
Automatically generate a constraint_value list.
"""

def _translate_cpu(arch):
    if arch in ["i386", "i486", "i586", "i686", "i786", "x86"]:
        return "x86_32"
    if arch in ["amd64", "x86_64", "x64"]:
        return "x86_64"
    if arch in ["ppc", "ppc64"]:
        return "ppc"
    if arch in ["ppc64le"]:
        return "ppc64le"
    if arch in ["arm", "armv7l"]:
        return "arm"
    if arch in ["aarch64"]:
        return "aarch64"
    if arch in ["s390x", "s390"]:
        return "s390x"
    if arch in ["mips64el", "mips64"]:
        return "mips64"
    if arch in ["riscv64"]:
        return "riscv64"
    return None

def _translate_os(os):
    if os.startswith("mac os"):
        return "osx"
    if os.startswith("freebsd"):
        return "freebsd"
    if os.startswith("openbsd"):
        return "openbsd"
    if os.startswith("linux"):
        return "linux"
    if os.startswith("windows"):
        return "windows"
    return None

def _translate_isa(arch):
    if arch in ["amd64", "x86_64", "x64"]:
        return "avx2"
    return None

def _has_cuda(rctx):
    if rctx.os.environ.get("CUDA_PATH", None):
        return True
    if rctx.which("ptxas"):
        return True
    if rctx.os.name.startswith("linux") and rctx.path("/usr/local/cuda").exists:
        return True
    return False

def _translate_device(rctx):
    if _has_cuda(rctx):
        return "cuda"
    return None

def _translate_compiler(rctx):
    if rctx.os.name.startswith("linux"):
        gcc_path = rctx.which("gcc")
        if gcc_path == None:
            return None

        gcc_version = rctx.execute([str(gcc_path), "-dumpversion"]).stdout.strip()
        if len(gcc_version) == 0:
            return None
        gcc_version = gcc_version.split(".")[0]

        # Only the gcc version defined in the aqinfer platform is processed.
        if gcc_version == "13" or gcc_version == "15":
            return "gcc%s" % gcc_version
    return None

def _host_platform_repo_impl(rctx):
    cpu = _translate_cpu(rctx.os.arch)
    os = _translate_os(rctx.os.name)
    isa = _translate_isa(rctx.os.arch)
    device = _translate_device(rctx)
    compiler = _translate_compiler(rctx)

    cpu = "" if cpu == None else "  '@platforms//cpu:%s',\n" % cpu
    os = "" if os == None else "  '@platforms//os:%s',\n" % os
    isa = "" if isa == None else "  '@tiny_llm//platforms/isa:%s',\n" % isa
    device = "" if device == None else "  '@tiny_llm//platforms/device:%s',\n" % device
    compiler = "" if compiler == None else "  '@tiny_llm//platforms/compiler_version:%s',\n" % compiler

    rctx.file("BUILD.bazel", """
# DO NOT EDIT: automatically generated BUILD file
""")

    rctx.file("constraints.bzl", """
# DO NOT EDIT: automatically generated constraints list
HOST_CONSTRAINS = [
%s%s%s%s%s]
""" % (cpu, os, isa, device, compiler))

host_platform_repo = repository_rule(
    implementation = _host_platform_repo_impl,
    doc = """Generates constraints for the host platform. The constraints.bzl
file contains a single <code>HOST_CONSTRAINS</code> variable, which is a
list of strings, each of which is a label to a <code>constraint_value</code>
for the host platform.""",
    local = True,
)

def _host_platform_impl(module_ctx):
    host_platform_repo(name = "host_platform")

host_platform = module_extension(
    implementation = _host_platform_impl,
    doc = """Generates a <code>host_platform_repo</code> repo named
<code>host_platform</code>, containing constraints for the host platform.""",
)
