package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*
  import Type.*

  def typeCheck(expr: Expr, tenv: TypeEnv): Type = expr match
    case Num(_)       => NumT
    case Bool(_)      => BoolT
    case Add(l, r)    => (typeCheck(l, tenv), typeCheck(r, tenv)) match
      case (NumT, NumT) => NumT
      case _ => error(s"Type error: expected numbers in addition, got ${l.str} and ${r.str}")
    case Mul(l, r)    => (typeCheck(l, tenv), typeCheck(r, tenv)) match
      case (NumT, NumT) => NumT
      case _ => error(s"Type error: expected numbers in multiplication, got ${l.str} and ${r.str}") 
    case Div(l, r)    => (typeCheck(l, tenv), typeCheck(r, tenv)) match
      case (NumT, NumT) => NumT
      case _ => error(s"Type error: expected numbers in division, got ${l.str} and ${r.str}")
    case Mod(l, r)    => (typeCheck(l, tenv), typeCheck(r, tenv)) match
      case (NumT, NumT) => NumT
      case _ => error(s"Type error: expected numbers in modulus, got ${l.str} and ${r.str}")
    case Eq(l, r)     => (typeCheck(l, tenv), typeCheck(r, tenv)) match
      case (NumT, NumT) => BoolT
      case (BoolT, BoolT) => BoolT
      case _ => error(s"Type error: expected numbers or booleans in equality, got ${l.str} and ${r.str}")
    case Lt(l, r)     => (typeCheck(l, tenv), typeCheck(r, tenv)) match
      case (NumT, NumT) => BoolT
      case _ => error(s"Type error: expected numbers in less-than comparison, got ${l.str} and ${r.str}")
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
    case If(c, t, e)  =>
      typeCheck(c, tenv) match
        case BoolT => 
          val tt = typeCheck(t, tenv)
          val et = typeCheck(e, tenv)
          if (tt == et) tt
          else error(s"Type error: expected then and else branches to have the same type, got ${tt.str} and ${et.str}")
        case _ => error(s"Type error: expected a boolean in if condition, got ${c.str}")
    case Rec(n, p, pt, rt, b, s) =>
      val funType = ArrowT(pt, rt)
      val tenvWithFun = tenv + (n -> funType)
      val bType = typeCheck(b, tenvWithFun + (p -> pt))
      if (bType == rt) typeCheck(s, tenvWithFun)
      else error(s"Type error: expected function body to have return type ${rt.str}, got ${bType.str}")

  def interp(expr: Expr, env: Env): Value = expr match
    case Num(n)       => NumV(n)
    case Bool(b)      => BoolV(b)
    case Add(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => NumV(n1 + n2)
      case _ => error(s"Runtime error: expected numbers in addition, got ${l.str} and ${r.str}")
    case Mul(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => NumV(n1 * n2)
      case _ => error(s"Runtime error: expected numbers in multiplication, got ${l.str} and ${r.str}")
    case Div(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => 
        if (n2 != 0) NumV(n1 / n2)
        else error(s"Runtime error: division by zero in ${l.str} / ${r.str}")
      case _ => error(s"Runtime error: expected numbers in division, got ${l.str} and ${r.str}")
    case Mod(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => 
        if (n2 != 0) NumV(n1 % n2)
        else error(s"Runtime error: modulus by zero in ${l.str} % ${r.str}")
      case _ => error(s"Runtime error: expected numbers in modulus, got ${l.str} and ${r.str}")
    case Eq(l, r)     => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => BoolV(n1 == n2)
      case (BoolV(b1), BoolV(b2)) => BoolV(b1 == b2)
      case _ => error(s"Runtime error: expected numbers or booleans in equality, got ${l.str} and ${r.str}")
    case Lt(l, r)     => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => BoolV(n1 < n2)
      case _ => error(s"Runtime error: expected numbers in less-than comparison, got ${l.str} and ${r.str}")
    case Val(x, i, b) => 
      val iv = interp(i, env)
      interp(b, env + (x -> iv))
    case Id(x)        => env.getOrElse(x, error(s"Runtime error: unbound identifier $x"))
    case Fun(p, t, b) => CloV(p, b, () => env)
    case App(f, e)    => 
      val fv = interp(f, env)
      val ev = interp(e, env)
      fv match
        case CloV(p, b, cloEnv) => 
          interp(b, cloEnv() + (p -> ev))
        case _ => error(s"Runtime error: expected a function in application, got ${f.str}")
    case If(c, t, e)  => 
      val cv = interp(c, env)
      cv match
        case BoolV(true) => interp(t, env)
        case BoolV(false) => interp(e, env)
        case _ => error(s"Runtime error: expected a boolean in if condition, got ${c.str}")
    case Rec(n, p, pt, rt, b, s) => 
      lazy val recEnv: Env = env + (n -> CloV(p, b, () => recEnv))
      interp(s, recEnv)

}
