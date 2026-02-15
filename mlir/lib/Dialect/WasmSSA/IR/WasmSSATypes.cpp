//===- WasmSSAOps.cpp - WasmSSA dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

namespace mlir::wasmssa {
#include "mlir/Dialect/WasmSSA/IR/WasmSSATypeConstraints.cpp.inc"

bool isWasmCompatibleValueType(Type type) {
  if (!type)
    return false;

  if (isWasmValueType(type))
    return true;

  Dialect &dialect = type.getDialect();

  auto *iface =
      dialect.getRegisteredInterface<WasmSSAValueTypeDialectInterface>();
  if (iface && iface->isWasmSSAExtendedValueType(type))
    return true;

  // Relaxed mode: allow non-builtin dialect types as wasmssa-compatible values.
  return dialect.getNamespace() != "builtin";
}
} // namespace mlir::wasmssa
