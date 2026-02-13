//===- WasmSSA.h - WasmSSA dialect ------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WasmSSA_IR_WasmSSA_H_
#define MLIR_DIALECT_WasmSSA_IR_WasmSSA_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// WebAssemblyDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSAOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// WebAssembly Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// WebAssembly Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h"

//===----------------------------------------------------------------------===//
// WebAssembly Dialect Operations
//===----------------------------------------------------------------------===//
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// WebAssembly Constraints
//===----------------------------------------------------------------------===//

namespace mlir::wasmssa {

/// Dialects may implement this interface to mark additional types as
/// WasmSSA-compatible value types.
class WasmSSAValueTypeDialectInterface
    : public DialectInterface::Base<WasmSSAValueTypeDialectInterface> {
public:
  using Base::Base;

  /// Returns true if `type` should be accepted as a value type by WasmSSA ops.
  virtual bool isWasmSSAExtendedValueType(Type type) const { return false; }
};

/// Returns true if `type` is either a built-in WasmSSA value type or a
/// type accepted by its defining dialect via WasmSSAValueTypeDialectInterface.
bool isWasmCompatibleValueType(Type type);

} // namespace mlir::wasmssa

namespace mlir {
namespace wasmssa {
#include "mlir/Dialect/WasmSSA/IR/WasmSSATypeConstraints.h.inc"
}
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOps.h.inc"

#endif // MLIR_DIALECT_WasmSSA_IR_WasmSSA_H_
