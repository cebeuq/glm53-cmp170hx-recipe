# Build-only image for the cmpunlocker + P2P (TRAP31) NVIDIA open kernel modules (x86_64 host).
# Match the host userland (Ubuntu 24.04 here) so the prebuilt kbuild helpers in the bind-mounted
# kernel headers run, and the compiler major version the kernel was built with (gcc-13 here;
# check CONFIG_CC_VERSION_TEXT in /lib/modules/$(uname -r)/build/.config). libdw: gendwarfksyms.
# Nothing here installs anything on the host.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      gcc-13 gcc g++-13 g++ make patch python3 python3-yaml kmod libelf1t64 libdw1t64 ca-certificates xz-utils file \
 && rm -rf /var/lib/apt/lists/*

COPY build-in-container.sh /usr/local/bin/build-in-container.sh
RUN chmod 0755 /usr/local/bin/build-in-container.sh
ENTRYPOINT ["/usr/local/bin/build-in-container.sh"]
