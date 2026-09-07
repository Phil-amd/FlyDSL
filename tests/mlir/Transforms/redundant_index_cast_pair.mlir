// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// A round trip through an intermediate at least 32 bits wide is redundant --
// but only when the value is provably narrow enough to survive it.  Values
// derived from launch coordinates and constants qualify.

// CHECK-LABEL: func.func @fold_from_thread_id
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_from_thread_id() -> index {
  %tid = gpu.thread_id x
  %c63 = arith.constant 63 : index
  %m = arith.andi %tid, %c63 : index
  %0 = arith.index_cast %m : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func.func @fold_arith_chain
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_arith_chain() -> index {
  %tid = gpu.thread_id x
  %bid = gpu.block_id x
  %c4 = arith.constant 4 : index
  %a = arith.muli %bid, %c4 : index
  %b = arith.addi %a, %tid : index
  %s = arith.shrui %b, %c4 : index
  %0 = arith.index_cast %s : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func.func @fold_i64_intermediate
// CHECK-NOT:     arith.index_cast
// CHECK:         return
func.func @fold_i64_intermediate() -> index {
  %tid = gpu.thread_id x
  %0 = arith.index_cast %tid : index to i64
  %1 = arith.index_cast %0 : i64 to index
  return %1 : index
}

// A value of unknown range must NOT be folded: an `index` is 64-bit, so a
// value above 2^31 would not survive the trip through i32.  This is the
// unsoundness llvm/llvm-project@8c81064169c5 fixed upstream.

// CHECK-LABEL: func.func @keep_unknown_arg
// CHECK:         arith.index_cast %arg0 : index to i32
// CHECK:         arith.index_cast %{{.*}} : i32 to index
func.func @keep_unknown_arg(%a: index) -> index {
  %0 = arith.index_cast %a : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// Likewise when an unknown value enters the arithmetic chain.

// CHECK-LABEL: func.func @keep_tainted_chain
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
func.func @keep_tainted_chain(%a: index) -> index {
  %tid = gpu.thread_id x
  %b = arith.addi %tid, %a : index
  %0 = arith.index_cast %b : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// A narrow intermediate truncates even a bounded value, so it is never folded.

// CHECK-LABEL: func.func @keep_i8
// CHECK:         arith.index_cast %{{.*}} : index to i8
// CHECK:         arith.index_cast %{{.*}} : i8 to index
func.func @keep_i8() -> index {
  %tid = gpu.thread_id x
  %0 = arith.index_cast %tid : index to i8
  %1 = arith.index_cast %0 : i8 to index
  return %1 : index
}

// CHECK-LABEL: func.func @keep_i16
// CHECK:         arith.index_cast %{{.*}} : index to i16
// CHECK:         arith.index_cast %{{.*}} : i16 to index
func.func @keep_i16() -> index {
  %tid = gpu.thread_id x
  %0 = arith.index_cast %tid : index to i16
  %1 = arith.index_cast %0 : i16 to index
  return %1 : index
}
