package kuplrg

object Implementation extends Template {

  import Expr.*

  def interp(expr: Expr, env: Env, fenv: FEnv): Value = expr match
    case Num(n) => n
    case Add(l,r) => interp(l,env,fenv) + interp(r,env,fenv)
    case Mul(l,r) => interp(l,env,fenv) * interp(r,env,fenv)
    case Val(x,e,b) => 
      val v = interp(e,env,fenv)
      interp(b,env + (x -> v),fenv)
    case Id(x) => env.getOrElse(x, error(s"free identifier: $x"))
    case App(fname, arg) =>
      // fenv에서 함수 정의를 찾음
      val fDef = fenv.getOrElse(fname, error(s"unknown function: $fname"))
      
      // 인자 식(arg)을 계산해서 값(Value)을 얻음
      val aVal = interp(arg, env, fenv)
      
      // 함수 본체(body)를 실행
      // 이때 환경(env)은 오직 이 함수의 매개변수(param)에 계산된 값(aVal)만 바인딩된 상태
      interp(fDef.body, Map(fDef.param -> aVal), fenv)


  def interpDS(expr: Expr, env: Env, fenv: FEnv): Value = expr match
    case Num(n) => n
    case Add(l,r) => interpDS(l,env,fenv) + interpDS(r,env,fenv)
    case Mul(l,r) => interpDS(l,env,fenv) * interpDS(r,env,fenv)
    case Val(x,e,b) => 
      val v = interpDS(e,env,fenv)
      interpDS(b,env + (x -> v),fenv)
    case Id(x) => env.getOrElse(x, error(s"free identifier: $x"))
    case App(fname, arg) =>
      // fenv에서 함수 정의를 찾음
      val fDef = fenv.getOrElse(fname, error(s"unknown function: $fname"))
      
      // 인자 식(arg)을 계산해서 값(Value)을 얻음
      val aVal = interpDS(arg, env, fenv)
      
      // 함수 본체(body)를 실행
      // 이때 환경(env)은 오직 이 함수의 매개변수(param)에 계산된 값(aVal)만 바인딩된 상태
      interpDS(fDef.body, env + (fDef.param -> aVal), fenv)
}
