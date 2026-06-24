package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*
  import Type.*

  def validType(ty: Type, tenv: TypeEnv): Type = ty match
    case NumT =>
      NumT
    case ArrowT(p, r) =>
      ArrowT(validType(p, tenv), validType(r, tenv))
    case VarT(name) =>
      if (!tenv.tys.contains(name)) error(s"unknown type: $name")
      VarT(name)
    case PolyT(name, ty) =>
      PolyT(name, validType(ty, tenv.addType(name)))
  
  def subst(bodyTy: Type, name: String, ty: Type): Type = bodyTy match
    case NumT => NumT
    case ArrowT(pty, rty) =>
      ArrowT(subst(pty, name, ty), subst(rty,name, ty))
    case VarT(x) => if (name == x) ty else VarT(x)
    case PolyT(x, bodyTy) =>
      if (name==x) PolyT(x, bodyTy)
      else PolyT(x, subst(bodyTy, name, ty))
  
  def isSame(lty:Type, rty:Type): Boolean = (lty, rty) match
    case (NumT, NumT) => true
    case (ArrowT(lpty, lrty),ArrowT(rpty, rrty)) =>
      isSame(lpty, rpty) && isSame(lrty, rrty)
    case (VarT(lname),VarT(rname) ) => lname == rname
    case (PolyT(lname, lty), PolyT(rname, rty)) =>
      isSame(lty, subst(rty, rname, VarT(lname)))
    case _ => false

  def typeCheck(expr: Expr, tenv: TypeEnv): Type = expr match
    case Num(number: BigInt) => NumT
  // additions
    case Add(left: Expr, right: Expr) => (typeCheck(left, tenv), typeCheck(right, tenv)) match
      case (NumT,NumT) => NumT
      case _ => error(s"Type error")
    // multiplications
    case Mul(left: Expr, right: Expr) => (typeCheck(left, tenv), typeCheck(right, tenv)) match
      case (NumT,NumT) => NumT
      case _ => error(s"Type error")
    // immutable variable definition
    case Val(name: String, init: Expr, body: Expr) =>
      val inty = typeCheck(init, tenv)
      validType(inty,tenv)
      typeCheck(body, tenv.addVar((name,inty)))
    // identifier lookups
    case Id(name: String) => tenv.vars.getOrElse(name, error(s"undefined variable: $name"))
    // anonymous (lambda) functions
    case Fun(param: String, ty: Type, body: Expr) => 
      validType(ty, tenv)
      val newtenv = tenv.addVar(param,ty)
      val bodyty = typeCheck(body, newtenv)
      ArrowT(ty, bodyty)

    // function applications
    case App(fun: Expr, arg: Expr) => 
      typeCheck(fun,tenv) match
        case ArrowT(inty, outty) => 
          val argty = typeCheck(arg, tenv)
          if (isSame(inty,argty)) outty else error(s"invalid type: ${inty.str}, ${argty.str}, ${outty.str}")
          
        case _ => error(s"")

    // type abstraction
    case TypeAbs(name: String, body: Expr) => 
      if (tenv.tys.contains(name)) error(s"already defined type: $name")
      PolyT(name, typeCheck(body, tenv.addType(name)))

    // type application
    case TypeApp(expr: Expr, ty: Type) => typeCheck(expr, tenv) match
      case PolyT(name, bodyty) => subst(bodyty, name, validType(ty, tenv))
      // name 안의 bodyty를 ty로 교체
      case t => error(s"not a polymorphic type: ${t.str}")

  def interp(expr: Expr, env: Env): Value = expr match
    case Num(number: BigInt) => NumV(number)
    // additions
    case Add(left: Expr, right: Expr) => 
      val lv = interp(left,env)
      val rv = interp(right,env)
      (lv, rv) match
        case (NumV(l),NumV(r)) => NumV(l+r)
        case (_,_) => error(s"invalid operators")
    // multiplications
    case Mul(left: Expr, right: Expr) =>
      val lv = interp(left,env)
      val rv = interp(right,env)
      (lv, rv) match
        case (NumV(l),NumV(r)) => NumV(l*r)
        case (_,_) => error(s"invalid operators")
    // immutable variable definition
    case Val(name: String, init: Expr, body: Expr) =>
      val inv = interp(init, env)
      interp(body, env + (name -> inv))
    // identifier lookups
    case Id(name: String) => env.getOrElse(name, error(s"invalid operators"))
    // anonymous (lambda) functions
    case Fun(param: String, ty: Type, body: Expr) =>
      CloV(param, body, env)
    // function applications
    case App(fun: Expr, arg: Expr) => interp(fun, env) match
      case CloV(param, body, fenv) => 
        val value = interp(arg, env)
        interp(body,fenv + (param -> value) )
      case _ => error(s"invalid operators")

      
    // type abstraction
    case TypeAbs(name: String, body: Expr) =>
      TypeAbsV(name, body, env)
    // type application
    case TypeApp(expr: Expr, ty: Type) => interp(expr, env) match
      case TypeAbsV(name, body, fenv) => interp(body, fenv)
      case v => error(s"not a type abstraction: ${v.str}")

}
