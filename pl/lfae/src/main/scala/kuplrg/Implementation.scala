package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*
  def force(v: Value): Value = v match
    case ExprV(e, eEnv) => force(interp(e, eEnv)) // 실제 값이 나올 때까지 재귀적으로 평가
    case _ => v // 이미 NumV나 CloV라면 그대로 반환

  def interp(expr: Expr, env: Env): Value = expr match
    case Num(n) => NumV(n)

    case Add(l, r) => 
      val lv = force(interp(l, env)) // l을 평가하고 force
      val rv = force(interp(r, env)) // r을 평가하고 force
      (lv, rv) match
        case (NumV(n1), NumV(n2)) => NumV(n1 + n2)
        case _ => error("invalid operation")
    
    case Mul(l, r) => 
      val lv = force(interp(l, env)) // l을 평가하고 force
      val rv = force(interp(r, env)) // r을 평가하고 force
      (lv, rv) match
        case (NumV(n1), NumV(n2)) => NumV(n1 * n2)
        case _ => error("invalid operation")
    
    case Id(x) =>
      env.getOrElse(x, error(s"free identifier: $x"))

    case Fun(param, body) =>
    // 클로저는 현재 환경을 저장합니다.
      CloV(param, body, env)
    
    case App(f, arg) =>
      val fv = force(interp(f, env)) // 함수 부분을 force해서 CloV를 얻음
      fv match
        case CloV(param, body, fEnv) =>
          // 인자(arg)는 지연 평가이므로 force하지 않고 ExprV로 넘김
          val thunk = ExprV(arg, env)
          interp(body, fEnv + (param -> thunk))
        case _ => error("not a function")
          
}
