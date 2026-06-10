package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*
  import Type.*

  def typeCheck(expr: Expr, tenv: TypeEnv): Type = expr match
    case Num(_)       => NumT
    case Add(l, r)    => (typeCheck(l, tenv), typeCheck(r, tenv)) match
      case (NumT, NumT) => NumT
      case _ => error(s"Type error: expected numbers in addition, got ${l.str} and ${r.str}")
    case Mul(l, r)    => (typeCheck(l, tenv), typeCheck(r, tenv)) match
      case (NumT, NumT) => NumT
      case _ => error(s"Type error: expected numbers in multiplication, got ${l.str} and ${r.str}") 
    case Val(x, i, b) => typeCheck(i, tenv) match
      case t => typeCheck(b, tenv + (x -> t))
    case Id(x)        => tenv.getOrElse(x, error(s"Type error: unbound identifier $x"))
    case Fun(p, t, b) => 
      val rt  = typeCheck(b, tenv + (p -> t))
      ArrowT(t, rt)
    case App(f, e)    => 
      val ft = typeCheck(f, tenv)
      ft match
        case ArrowT(pt, rt) => 
          val et = typeCheck(e, tenv)
          if (et == pt) rt
          else error(s"Type error: expected argument of type ${pt.str}, got ${et.str}")
        case _ => error(s"Type error: expected a function in application, got ${f.str}")

  def interp(expr: Expr, env: Env): Value = expr match
    case Num(n)       => NumV(n)
    case Add(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => NumV(n1 + n2)
      case _ => error(s"Runtime error: expected numbers in addition, got ${l.str} and ${r.str}")
    case Mul(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => NumV(n1 * n2)
      case _ => error(s"Runtime error: expected numbers in multiplication, got ${l.str} and ${r.str}")
    case Val(x, i, b) => 
      val iv = interp(i, env)
      interp(b, env + (x -> iv))
    case Id(x)        => env.getOrElse(x, error(s"Runtime error: unbound identifier $x"))
    case Fun(p, t, b) => CloV(p, b, env)
    case App(f, e)    => 
      val fv = interp(f, env)
      val ev = interp(e, env)
      fv match
        case CloV(p, b, cloEnv) => 
          interp(b, cloEnv + (p -> ev))
        case _ => error(s"Runtime error: expected a function in application, got ${f.str}")

}
