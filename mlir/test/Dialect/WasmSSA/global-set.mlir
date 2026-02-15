// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -verify-diagnostics --split-input-file

module {
  wasmssa.global @g i32 mutable : {
    %c0 = wasmssa.const 0 : i32
    wasmssa.return %c0 : i32
  }

  wasmssa.func @set_it() {
    %v = wasmssa.const 7 : i32
    wasmssa.global_set @g to %v : i32
    wasmssa.return
  }
}

// CHECK-LABEL: wasmssa.func @set_it
// CHECK: %[[V:.*]] = wasmssa.const 7 : i32
// CHECK: wasmssa.global_set @g to %[[V]] : i32

// -----

module {
  wasmssa.func @f() {
    %v = wasmssa.const 1 : i32
    // expected-error @+1 {{symbol @missing is undefined}}
    wasmssa.global_set @missing to %v : i32
    wasmssa.return
  }
}

// -----

module {
  wasmssa.global @g i32 : {
    %c0 = wasmssa.const 0 : i32
    wasmssa.return %c0 : i32
  }

  wasmssa.func @f() {
    %v = wasmssa.const 1 : i32
    // expected-error @+1 {{global.set target must be mutable}}
    wasmssa.global_set @g to %v : i32
    wasmssa.return
  }
}

// -----

module {
  wasmssa.global @g i64 mutable : {
    %c0 = wasmssa.const 0 : i64
    wasmssa.return %c0 : i64
  }

  wasmssa.func @f() {
    %v = wasmssa.const 1 : i32
    // expected-error @+1 {{global.set value type does not match target global type: expected 'i64' but got 'i32'}}
    wasmssa.global_set @g to %v : i32
    wasmssa.return
  }
}

// -----

module {
  wasmssa.func @not_global() {
    wasmssa.return
  }

  wasmssa.func @f() {
    %v = wasmssa.const 1 : i32
    // expected-error @+1 {{symbol @not_global is not a global symbol}}
    wasmssa.global_set @not_global to %v : i32
    wasmssa.return
  }
}
