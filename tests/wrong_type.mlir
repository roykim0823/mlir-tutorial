func.func @bad(%a: i32, %b: i64) -> i32 {
  %0 = arith.muli %a, %b : i32
  func.return %0 : i32
}
