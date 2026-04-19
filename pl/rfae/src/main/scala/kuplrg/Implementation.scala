package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*

  def interp(expr: Expr, env: Env): Value = expr match
    case Num(n) => NumV(n)
    case Bool(b) => BoolV(b)
    case Id(x) => env.getOrElse(x, error(s"free identifier: $x"))
    case Add(l, r) => 
      (interp(l, env), interp(r, env)) match
        case (NumV(n1), NumV(n2)) => NumV(n1 + n2)
        case (v1, v2) => error(s"invalid operation: ${v1.str} + ${v2.str}")

    case Mul(l, r) => 
      (interp(l, env), interp(r, env)) match
        case (NumV(n1), NumV(n2)) => NumV(n1 * n2)
        case (v1, v2) => error(s"invalid operation: ${v1.str} * ${v2.str}")
    
    case Div(l, r) => 
      (interp(l, env), interp(r, env)) match
        case (NumV(n1), NumV(n2)) =>
          if (n2 == 0) error(s"invalid operation")
          else NumV(n1 / n2)
        case (v1, v2) => error(s"invalid operation: ${v1.str} * ${v2.str}")
    
    case Mod(l, r) => 
      (interp(l, env), interp(r, env)) match
        case (NumV(n1), NumV(n2)) => 
          if (n2 == 0) error(s"invalid operation")
          else NumV(n1 % n2)
        case (v1, v2) => error(s"invalid operation: ${v1.str} * ${v2.str}")
    
    case Eq(l, r) => 
      (interp(l, env), interp(r, env)) match
        case (NumV(n1), NumV(n2)) => if (n1==n2) BoolV(true) else BoolV(false)
        case (v1, v2) => error(s"invalid operation: ${v1.str} * ${v2.str}")
    
    case Lt(l, r) => 
      (interp(l, env), interp(r, env)) match
        case (NumV(n1), NumV(n2)) => if (n1<n2) BoolV(true) else BoolV(false)
        case (v1, v2) => error(s"invalid operation: ${v1.str} * ${v2.str}")

    case Fun(param, body) =>
      CloV(param, body, () => env)
    
    case Rec(n,p,b,s) =>
      lazy val newEnv: Env = env + (n -> CloV(p, b, () => newEnv))
      interp(s, newEnv)

    case App(fExpr, argExpr) => 
      val fVal = interp(fExpr, env)
      fVal match {
        case CloV(p, b, fEnv) => interp(b, fEnv() + (p -> interp(argExpr, env)))

        case v => error(s"not a function: ${v.str}")
      }
    
    case If(c, t, e) => interp(c, env) match
      case BoolV(true) => interp(t, env)
      case BoolV(false) => interp(e, env)
      case v => error(s"not a boolean: ${v.str}")
}
