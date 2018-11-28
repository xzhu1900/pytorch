#pragma once

namespace torch {
namespace data {
namespace ctf {

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete

#define DISABLE_COPY_AND_MOVE(TypeName)          \
  TypeName(const TypeName&) = delete;            \
  TypeName& operator=(const TypeName&) = delete; \
  TypeName(TypeName&&) = delete;                 \
  TypeName& operator=(TypeName&&) = delete

} // namespace ctf
} // namespace data
} // namespace torch