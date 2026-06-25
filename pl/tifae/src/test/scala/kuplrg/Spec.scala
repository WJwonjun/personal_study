package kuplrg

import Implementation.*

class Spec extends SpecBase {

  // -------------------------------------------------------------------------
  // eval
  // -------------------------------------------------------------------------
  test(eval("42"), "42: Number")
  test(eval("1 + 2"), "3: Number")
  test(eval("2 * 3"), "6: Number")
  test(eval("1 + 2 * 3"), "7: Number")
  // free identifier: x
  testExc(eval("x"))
  test(eval("val x = 42; x"), "42: Number")
  test(eval("val x = 1; val y = 2; x + y"), "3: Number")
  test(eval("x => x"), "<function>: [T1] { T1 => T1 }")
  test(eval("x => x + 1"), "<function>: Number => Number")
  test(eval("val f = x => x && x; f"), "<function>: Boolean => Boolean")
  // not a function: Number
  testExc(eval("1(2)"))
  // not a function: Number
  test(eval("x => x(1)"), "<function>: [T1] { (Number => T1) => T1 }")
  test(eval("val f = x => x; f(42)"), "42: Number")
  test(
    eval(
      "x => y => z => if (x) y else z",
    ),
    "<function>: [T1] { Boolean => T1 => T1 => T1 }",
  )
  test(eval("def f(x) = x; f"), "<function>: [T1] { T1 => T1 }")
  test(
    eval(
      "def sum(x) = if (x < 1) 0 else x + sum(x - 1); sum",
    ),
    "<function>: Number => Number",
  )
  // cyclic type
  testExc(eval("def f(x) = if (true) x(2) else x; f"))
  // cyclic type
  testExc(eval("x => x(1)(x)"))
  // Cannot unify T5 => T5 and Number
  testExc(eval("(x => x(y => y)(x(1)))(z => z)"))
  test(eval("val x = z => z; x(y => y)(x(1))"), "1: Number")

  val expr1 = """
    x => {
      val y = x + 1;
      z => z + y * z
    }
  """
  test(
    eval(expr1),
    "<function>: Number => Number => Number",
    weight = 3,
  )

  val expr2 = """
    f => {
      val x = f(1) + f(2);
      f(f(f(x)))
    }
  """
  test(
    eval(expr2),
    "<function>: (Number => Number) => Number",
    weight = 3,
  )

  val expr3 = """
    f => {
      val g = f(1);
      val h = f(2);
      g(h(f(3)(4)))
    }
  """
  test(
    eval(expr3),
    "<function>: (Number => Number => Number) => Number",
    weight = 3,
  )

  val expr4 = """
    f => {
      val g = x => x + 1;
      val h = f(g);
      f(f(f(h)))(0)
    }
  """
  test(
    eval(expr4),
    "<function>: ((Number => Number) => Number => Number) => Number",
    weight = 3,
  )

  val expr5 = """
    f => {
      val g = x => x + 1;
      val h = f(g);
      f(x => x + 1)(3) + h(0)
    }
  """
  test(
    eval(expr5),
    "<function>: ((Number => Number) => Number => Number) => Number",
    weight = 3,
  )

  val expr6 = """
    def f(a) = b => c => {
      if (b == 1) {
        if (a(0)) c(1)
        else c(b)
      } else {
        if (a(b)) c(b)
        else f(a)(b - 1)(c)
      }
    }
    f
  """
  test(
    eval(expr6),
    "<function>: [T1] { (Number => Boolean) => Number => (Number => T1) => T1 }",
    weight = 3,
  )

  val expr7 = """
    def f(a) = b => c => {
      if (a(1) == 3) b - 1
      else if (c(5) < a(0)) 0
      else f(a)(b * 2)(c)
    }
    f
  """
  test(
    eval(expr7),
    "<function>: (Number => Number) => Number => (Number => Number) => Number",
    weight = 3,
  )

  val expr8 = """
    val f = x => x
    val x = f(1)
    val y = f(true)
    val g = f(x => x + 1)
    if (y) g(2) else g(3)
  """
  test(
    eval(expr8),
    "3: Number",
    weight = 3,
  )

  // cyclic type
  val expr9 = """
    val mkRec = body => {
      val fX = fY => {
        val f = x => fY(fY)(x);
        body(f)
      }
      fX(fX)
    }
    mkRec
  """
  testExc(eval(expr9), weight = 3)

  val expr10 = """
    def gcd(x) = y => {
      if (x == 0) y
      else gcd(y % x)(x)
    }
    def twice(f) = x => f(f(x))
    def identity(x) = x
    twice(identity)(gcd(3)(5))
  """
  test(eval(expr10), "1: Number", weight = 3)
  /* Write your own tests */

}
