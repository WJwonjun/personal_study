package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*

  def interp(expr: Expr, env: Env): Value = expr match
    case Num(n) => NumV(n)
    case Add(l, r) => 
      (interp(l, env), interp(r, env)) match
        case (NumV(n1), NumV(n2)) => NumV(n1 + n2)
        case (v1, v2) => error(s"invalid operation: ${v1.str} + ${v2.str}")
    case Mul(l, r) => 
      (interp(l, env), interp(r, env)) match
        case (NumV(n1), NumV(n2)) => NumV(n1 * n2)
        case (v1, v2) => error(s"invalid operation: ${v1.str} * ${v2.str}")
    case Id(x) => env.getOrElse(x, error(s"free identifier: $x"))
    case Fun(param, body) => CloV(param, body, env)
    case App(fExpr, argExpr) => 
      val fVal = interp(fExpr, env)
      fVal match {
        case CloV(p, b, fEnv) =>
          val aVal = interp(argExpr, env)
          // [핵심] fEnv(정의 당시 환경) + 인자값으로 본체를 실행합니다.
          interp(b, fEnv + (p -> aVal))

        case _ => 
          error(s"not a function: ${fVal.str}")
      }

  def interpDS(expr: Expr, env: Env): Value = expr match
    case Num(n) => NumV(n)
    case Add(l, r) => 
      (interpDS(l, env), interpDS(r, env)) match
        case (NumV(n1), NumV(n2)) => NumV(n1 + n2)
        case (v1, v2) => error(s"invalid operation: ${v1.str} + ${v2.str}")
    case Mul(l, r) => 
      (interpDS(l, env), interpDS(r, env)) match
        case (NumV(n1), NumV(n2)) => NumV(n1 * n2)
        case (v1, v2) => error(s"invalid operation: ${v1.str} * ${v2.str}")
    case Id(x) => env.getOrElse(x, error(s"free identifier: $x"))
    case Fun(param, body) => CloV(param, body, env)
    case App(fExpr, argExpr) => 
      val fVal = interpDS(fExpr, env)
      fVal match {
        case CloV(p, b, fEnv) =>
          val aVal = interpDS(argExpr, env)
          // [핵심] fEnv(정의 당시 환경) + 인자값으로 본체를 실행합니다.
          interpDS(b, env + (p -> aVal))

        case _ => 
          error(s"not a function: ${fVal.str}")
      }
}
