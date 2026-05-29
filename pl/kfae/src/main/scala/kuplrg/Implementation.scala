package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*
  import Cont.*
  def numAdd(l: Value, r: Value) : Value = (l, r) match
    case (NumV(lv), NumV(rv)) => NumV(lv+rv)
    case (_,_) => error(s"invalid operation")
  
  def numMul(l: Value, r: Value) : Value = (l, r) match
    case (NumV(lv), NumV(rv)) => NumV(lv*rv)
    case (_,_) => error(s"invalid operation")

  def reduce(k: Cont, s: Stack): (Cont, Stack) = (k, s) match
    case (EvalK(env, expr, k), s) => expr match
      case Num(n) => (k, NumV(n)::s)
      case Add(l, r) => (EvalK(env, l, EvalK(env, r, AddK(k))), s)
      case Mul(l, r) => (EvalK(env, l, EvalK(env, r, MulK(k))), s)
      case Id(x) => (env.get(x) match
        case Some(v) => (k, v::s)
        case None => error(s"free identifier: $x")
      )
      case Fun(p,b) => (k, CloV(p,b,env)::s)
      case App(f,a) => (EvalK(env, f, EvalK(env, a, AppK(k))), s)
      case Vcc(x, b) => (EvalK(env + (x -> ContV(k, s)), b, k), s)

    case (AddK(k), r::l::s) => (k, numAdd(l,r)::s)
    case (MulK(k), r::l::s) => (k, numMul(l,r)::s)
    case (AppK(k), a::f::s) => f match
      case CloV(p,b,fenv) => (EvalK(fenv + (p -> a), b, k), s)
      case ContV(k2, s2) => (k2, a::s2)
      case v => error(s"not a function: ${v.str}")

    case (EmptyK, s) => (EmptyK,s)
}
