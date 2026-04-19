package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*

  def interp(expr: Expr, env: Env, mem: Mem): (Value, Mem) = expr match
    case Num(n) => (NumV(n), mem)

    case Add(l, r) => 
      val (lv, lmem) = interp(l, env, mem)
      val (rv, rmem) = interp(r, env, lmem)
      (lv, rv) match
        case (NumV(n1), NumV(n2)) => (NumV(n1 + n2), rmem)
        case (v1, v2) => error(s"invalid operation: ${v1} + ${v2}")

    case Mul(l, r) => 
      val (lv, lmem) = interp(l, env, mem)
      val (rv, rmem) = interp(r, env, lmem)
      (lv, rv) match
        case (NumV(n1), NumV(n2)) => (NumV(n1 * n2), rmem)
        case (v1, v2) => error(s"invalid operation: ${v1} * ${v2}")
    
    case Id(x) => (env.getOrElse(x, error(s"free identifier: $x")), mem)

    case Fun(param, body) => (CloV(param, body, env), mem)

    case App(fun: Expr, arg: Expr) => 
      val (obj, nmem) = interp(fun, env, mem) 
      obj match
        case CloV(p, b, cloenv) => 
          val (x,y) = interp(arg, env, nmem)
          interp(b, cloenv + (p -> x), y)
        case _ => error("not a function")

    case NewBox(content: Expr) =>
      val (cv, cmem) = interp(content, env, mem)
      val addr = mem.keySet.maxOption.fold(0)(_ + 1)
      (BoxV(addr),cmem + (addr -> cv))
    
    case GetBox(b) =>
      val (bv, bmem) = interp(b, env, mem)
      bv match
      case BoxV(addr) => (bmem(addr), bmem)
      case _ => error(s"not a box: ${bv.str}")
    
    case SetBox(b, c) =>
      val (bv, bmem) = interp(b, env, mem)
      bv match
      case BoxV(addr) =>
        val (cv, cmem) = interp(c, env, bmem)
        (cv, cmem + (addr -> cv))
      case _ =>
        error(s"not a box: ${bv.str}")
    
    case Seq(l, r) =>
      val (_, lmem) = interp(l, env, mem)
      interp(r, env, lmem)
}
