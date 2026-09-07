// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// A round trip through an intermediate at least as wide as the values FlyDSL
// routes through this path is redundant and is folded away.

// CHECK-LABEL: func.func @fold_i32
// CHECK-NOT:     arith.index_cast
// CHECK:         return %arg0
func.func @fold_i32(%a: index) -> index {
  %0 = arith.index_cast %a : index to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func.func @fold_i64
// CHECK-NOT:     arith.index_cast
// CHECK:         return %arg0
func.func @fold_i64(%a: index) -> index {
  %0 = arith.index_cast %a : index to i64
  %1 = arith.index_cast %0 : i64 to index
  return %1 : index
}

// A narrow intermediate truncates, so the round trip is NOT redundant and must
// be preserved -- this is the unsoundness llvm/llvm-project@8c81064169c5 fixed
// upstream, and re-folding it here would reintroduce it.

// CHECK-LABEL: func.func @keep_i8
// CHECK:         arith.index_cast %arg0 : index to i8
// CHECK:         arith.index_cast %{{.*}} : i8 to index
func.func @keep_i8(%a: index) -> index {
  %0 = arith.index_cast %a : index to i8
  %1 = arith.index_cast %0 : i8 to index
  return %1 : index
}

// CHECK-LABEL: func.func @keep_i16
// CHECK:         arith.index_cast %arg0 : index to i16
// CHECK:         arith.index_cast %{{.*}} : i16 to index
func.func @keep_i16(%a: index) -> index {
  %0 = arith.index_cast %a : index to i16
  %1 = arith.index_cast %0 : i16 to index
  return %1 : index
}

// An i32 -> index -> i32 chain is a different shape (the outer type is not
// `index`) and is left to the upstream canonicalizer.

// CHECK-LABEL: func.func @not_our_shape
// CHECK:         arith.index_cast
func.func @not_our_shape(%a: i32) -> i32 {
  %0 = arith.index_cast %a : i32 to index
  %1 = arith.shrui %0, %0 : index
  %2 = arith.index_cast %1 : index to i32
  return %2 : i32
}
