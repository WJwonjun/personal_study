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

    case Var(name: String, init: Expr, body: Expr) =>
      val (cv, cmem) = interp(init, env, mem)
      val addr = cmem.keySet.maxOption.fold(0)(_ + 1)
      interp(body, env + (name -> addr), cmem + (addr -> cv) )

    case Id(x) => 
      val addr = env.getOrElse(x, error(s"free identifier: $x"))
      val value = mem.getOrElse(addr, error(s"unallocated address: $addr"))
      (value, mem)


    case Fun(param, body) => 
      (CloV(param, body, env), mem)


    case App(f, arg) =>
      val (fv, m1) = interp(f, env, mem)
      val (av, m2) = interp(arg, env, m1)
      fv match {
        case CloV(param, body, fEnv) =>
          val addr = m2.keySet.maxOption.fold(0)(_ + 1)
          interp(body, fEnv + (param -> addr), m2 + (addr -> av))

        case _ => error(s"not a function: $fv")
      }

    case Assign(x,e) =>
      val (eVal, eMem) = interp(e, env, mem)
      val addr = env.getOrElse(x, error(s"free identifier: $x"))
      (eVal, eMem + (addr -> eVal))

    case Seq(l,r) =>
      val (lVal, lMem) = interp(l,env,mem)
      interp(r,env,lMem)

  def interpCBR(expr: Expr, env: Env, mem: Mem): (Value, Mem) = expr match
    case Num(n) => (NumV(n), mem)

    case Add(l, r) => 
      val (lv, lmem) = interpCBR(l, env, mem)
      val (rv, rmem) = interpCBR(r, env, lmem)
      (lv, rv) match
        case (NumV(n1), NumV(n2)) => (NumV(n1 + n2), rmem)
        case (v1, v2) => error(s"invalid operation: ${v1} + ${v2}")

    case Mul(l, r) => 
      val (lv, lmem) = interpCBR(l, env, mem)
      val (rv, rmem) = interpCBR(r, env, lmem)
      (lv, rv) match
        case (NumV(n1), NumV(n2)) => (NumV(n1 * n2), rmem)
        case (v1, v2) => error(s"invalid operation: ${v1} * ${v2}")

    case Var(name: String, init: Expr, body: Expr) =>
      val (cv, cmem) = interpCBR(init, env, mem)
      val addr = cmem.keySet.maxOption.fold(0)(_ + 1)
      interpCBR(body, env + (name -> addr), cmem + (addr -> cv) )

    case Id(x) => 
      val addr = env.getOrElse(x, error(s"free identifier: $x"))
      val value = mem.getOrElse(addr, error(s"unallocated address: $addr"))
      (value, mem)


    case Fun(param, body) => 
      (CloV(param, body, env), mem)


    case App(f, arg) =>
      // 1. 함수(f)를 먼저 평가해서 클로저를 얻습니다.
      val (fv, m1) = interpCBR(f, env, mem)
      
      fv match
        case CloV(param, body, fEnv) =>
          arg match
            // 인자가 변수(Id)인 경우 -> 주소를 그대로 전달 (CBR의 핵심!)
            case Id(x) =>
              val addr = env.getOrElse(x, error(s"free identifier: $x"))
              interpCBR(body, fEnv + (param -> addr), m1)

            // 인자가 변수가 아닌 식인 경우 -> 계산 후 새로운 주소에 할당
            case _ =>
              val (av, m2) = interpCBR(arg, env, m1)
              val newAddr = m2.keySet.maxOption.fold(0)(_ + 1)
              interpCBR(body, fEnv + (param -> newAddr), m2 + (newAddr -> av))
        
        case _ => error(s"not a function: $fv")

    case Assign(x,e) =>
      val (eVal, eMem) = interpCBR(e, env, mem)
      val addr = env.getOrElse(x, error(s"free identifier: $x"))
      (eVal, eMem + (addr -> eVal))

    case Seq(l,r) =>
      val (lVal, lMem) = interpCBR(l,env,mem)
      interpCBR(r,env,lMem)
}
