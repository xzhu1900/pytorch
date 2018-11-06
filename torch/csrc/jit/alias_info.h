#pragma once
#include <unordered_set>
#include <vector>
#include "torch/csrc/jit/interned_strings.h"

namespace torch {
namespace jit {

class AliasInfo {
 public:
  // Symbol for the set that can alias
  static Symbol wildcard() {
    static const Symbol wc = Symbol::fromQualString("alias::*");
    return wc;
  }

  AliasInfo() {}

  void setIsWrite(bool isWrite) {
    isWrite_ = isWrite;
  }
  bool isWrite() const {
    return isWrite_;
  }

  void addSet(Symbol aliasSet) {
    sets_.insert(aliasSet);
  }
  // At the beginning of this op, which alias sets does this value belong to?
  const std::unordered_set<Symbol>& sets() const {
    return sets_;
  }
  // the alias info for the contained types of the type
  // e.g. if this is an annotation on List[T], `sets` refers to
  // the alias sets that the list may be in
  // while containedTypes()[0] refers to the sets that members of the list
  // may be in
  void addContainedType(AliasInfo aliasInfo) {
    containedTypes_.push_back(std::move(aliasInfo));
  }
  const std::vector<AliasInfo>& containedTypes() const {
    return containedTypes_;
  }

 private:
  std::unordered_set<Symbol> sets_;
  std::vector<AliasInfo> containedTypes_;
  bool isWrite_ = false;
};

} // namespace jit
} // namespace torch
