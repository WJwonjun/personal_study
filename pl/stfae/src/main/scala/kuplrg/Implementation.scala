package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*
  import Type.*

  def isSame(lty:Type, rty:Type): Boolean = (lty, rty) match
    case (NumT, NumT) => true
    case (ArrowT(lpty, lrty),ArrowT(rpty, rrty)) =>
      isSame(lpty, rpty) && isSame(lrty, rrty)
    case (_,_) => false

  def isSubtype(ltype: Type, rtype: Type): Boolean = (ltype, rtype) match
    case (BotT, rtype) => true
    case (ltype, TopT) => true
    case (RecordT(lfields), RecordT(rfields)) =>
      rfields.forall { case (key, rty) =>
        lfields.get(key) match
          case Some(lty) => isSubtype(lty, rty)
          case None => false
      }
    case (ArrowT(lp,lb), ArrowT(rp,rb)) => isSubtype(rp,lp) && isSubtype(lb,rb)
    case (_,_) => isSame(ltype,rtype)

  def typeCheck(expr: Expr, tenv: TypeEnv): Type = expr match
    case Num(number: BigInt) => NumT
    // additions
    case Add(left: Expr, right: Expr) => (typeCheck(left, tenv), typeCheck(right, tenv)) match
      case (NumT,NumT) => NumT
      case (NumT,BotT) => NumT
      case (BotT,NumT) => NumT
      case _ => error(s"Type error_Add")
    // multiplications
    case Mul(left: Expr, right: Expr) => (typeCheck(left, tenv), typeCheck(right, tenv)) match
      case (NumT,NumT) => NumT
      case (NumT,BotT) => NumT
      case (BotT,NumT) => NumT
      case _ => error(s"Type error_Mul")
    // immutable variable definition
    case Val(name: String, tyOpt: Option[Type], init: Expr, body: Expr) => tyOpt match
      case Some(t) =>
        val inty = typeCheck(init, tenv)
        if (isSubtype(inty, t)) typeCheck(body, tenv + (name -> t)) else error(s"invalid type")
      case _ =>
        val inty = typeCheck(init, tenv)
        typeCheck(body, tenv + (name -> inty))

      
    // identifier lookups
    case Id(name: String) => tenv.getOrElse(name, error(s"invalid type"))
    // anonymous (lambda) functions
    case Fun(param: String, ty: Type, body: Expr) => 
      val newtenv = tenv + (param -> ty)
      val bodyty = typeCheck(body, newtenv)
      ArrowT(ty, bodyty)
    // function applications
    case App(fun: Expr, arg: Expr) => 
      typeCheck(fun,tenv) match
        case ArrowT(inty, outty) => 
          val argty = typeCheck(arg, tenv)
          if (isSubtype(argty, inty)) outty else error(s"invalid type: ${inty.str}, ${argty.str}, ${outty.str}")
          
        case _ => error(s"")
    // records
    case Record(fields: List[(String, Expr)]) =>
      RecordT(fields.map { case (f, e) => (f, typeCheck(e, tenv)) }.toMap)
    // field lookups
    case Access(record: Expr, field: String) => typeCheck(record, tenv) match
      case RecordT(fs) => fs.getOrElse(field, error("no such field"))
      case _ => error("not a record")
    case Exit => BotT

  def interp(expr: Expr, env: Env): Value = expr match
    case Num(number: BigInt) => NumV(number)
    // additions
    case Add(left: Expr, right: Expr) => (interp(left, env), interp(right, env)) match
      case (NumV(l),NumV(r)) => NumV(l+r)
      case (_,_) => error("invalid operators_add")
    // multiplications
    case Mul(left: Expr, right: Expr) => (interp(left, env), interp(right, env)) match
      case (NumV(l),NumV(r)) => NumV(l*r)
      case (_,_) => error("invalid operators_mul")
    // immutable variable definition
    case Val(name: String, tyOpt: Option[Type], init: Expr, body: Expr) => 
      val inv = interp(init,env)
      val newenv = env + (name -> inv)
      interp(body, newenv)

    // identifier lookups
    case Id(name: String) => env.getOrElse(name, error("invalid_id"))
    // anonymous (lambda) functions
    case Fun(param: String, ty: Type, body: Expr) =>
      CloV(param, body,env)
    // function applications
    case App(fun: Expr, arg: Expr) => interp(fun,env) match 
      case CloV(param, body, fenv) =>
        val argval = interp(arg, env)
        val newenv = fenv + (param -> argval)
        interp(body, newenv)
    // records
    case Record(fields: List[(String, Expr)]) => 
      RecordV(
        fields.map{case (s,e) => (s, interp(e, env)) }.toMap
      )
    // field lookups
    case Access(record: Expr, field: String) => interp(record, env) match
      case RecordV(fields) => fields.getOrElse(field, error("invalid access"))
    // exit
    case Exit => error("Program terminated")

}
