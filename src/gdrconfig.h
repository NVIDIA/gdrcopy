#pragma once

#if defined __GNUC__
#if defined(__powerpc__)
#define GDRAPI_POWER
#elif defined(__aarch64__)
#define GDRAPI_ARM64
#elif defined(__i386__) || defined(__x86_64__) || defined(__X86__)
#define GDRAPI_X86
#else
#error "architecture is not supported"
#endif // arch
#else
#error "compiler not supported"
#endif // __GNUC__
