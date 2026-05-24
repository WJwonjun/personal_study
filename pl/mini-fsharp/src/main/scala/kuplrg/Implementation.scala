package kuplrg

object Implementation extends Template {

  import Expr.*, Value.*, Pattern.*

  // ---------------------------------------------------------------------------
  // Problem #1
  // ---------------------------------------------------------------------------
  def matchPattern(pattern: Pattern, v: Value): Option[Env] = (pattern, v) match
    case (PId(name), _) => Some(Map(name -> v))
    case (PNum(n), NumV(m)) if n == m => Some(Map())
    case (PBool(b1), BoolV(b2)) if b1 == b2 => Some(Map())
    case (PNil, ListV(Nil)) => Some(Map())
    case (PNone, NoneV) => Some(Map())
    case (PSome(p), SomeV(inner)) => matchPattern(p, inner)
    case (PCons(headP, tailP), ListV(head :: tail)) =>
      for
        hEnv <- matchPattern(headP, head)
        tEnv <- matchPattern(tailP, ListV(tail))
      yield hEnv ++ tEnv
    case (PTuple(patterns), TupleV(values)) if patterns.length == values.length =>
      patterns.zip(values).foldLeft(Option(Map.empty[String, Value])) {
        case (Some(acc), (p, v)) => matchPattern(p, v).map(acc ++ _)
        case _ => None
      }
    case _ => None

  def interp(expr: Expr, env: Env): Value = expr match
    case ENum(number: BigInt) => NumV(number)
    // booleans
    case EBool(bool: Boolean) => BoolV(bool)
    // identifier lookups
    case EId(name: String) => env.getOrElse(name, error(s"free identifier: $name"))
    // negation
    case ENeg(expr: Expr) => interp(expr, env) match
      case NumV(n) => NumV(n*(-1))
      case BoolV(b) => BoolV(!b)
      case _ => error("invalid operation")
    // addition
    case EAdd(left: Expr, right: Expr) =>
      val lVal = interp(left, env)
      val rVal = interp(right, env)
      (lVal, rVal) match
        case (NumV(l), NumV(r)) => NumV(l+r)
        case _ => error("invalid operation")
    // multiplication
    case EMul(left: Expr, right: Expr) =>
      val lVal = interp(left, env)
      val rVal = interp(right, env)
      (lVal, rVal) match
        case (NumV(l), NumV(r)) => NumV(l*r)
        case _ => error("invalid operation")
    // division
    case EDiv(left: Expr, right: Expr) =>
      val lVal = interp(left, env)
      val rVal = interp(right, env)
      (lVal, rVal) match
        case (NumV(l), NumV(0)) => error("invalid operation")
        case (NumV(l), NumV(r)) => NumV(l/r)
        case _ => error("invalid operation")
    // modulo
    case EMod(left: Expr, right: Expr) =>

      val lVal = interp(left, env)
      val rVal = interp(right, env)
      (lVal, rVal) match
        case (NumV(l), NumV(0)) => error("invalid operation")
        case (NumV(l), NumV(r)) => NumV(l%r)
        case (_,_) => error("invalid operation")
    // equal-to
    case EEq(left: Expr, right: Expr) =>
      val lVal = interp(left, env)
      val rVal = interp(right, env)
      (lVal, rVal) match
        case (_, _) => if (lVal==rVal) BoolV(true) else BoolV(false)

    // less-than
    case ELt(left: Expr, right: Expr) =>
      val lVal = interp(left, env)
      val rVal = interp(right, env)
      (lVal, rVal) match
        case (NumV(l), NumV(r)) => if (l<r) BoolV(true) else BoolV(false)
        case (_,_) => error("invalid operation")
    // conditional
    case EIf(cond: Expr, thenExpr: Expr, elseExpr: Expr) =>
      val conVal = interp(cond, env)
      conVal match
        case (BoolV(true)) => interp(thenExpr,env)
        case (BoolV(false)) => interp(elseExpr, env)
        case _ => error("not a boolean")
    // empty list
    case ENil => ListV(List())
    // list cons
    case ECons(head: Expr, tail: Expr) =>
      val headVal = interp(head, env)
      val tailVal = interp(tail, env)
      tailVal match
        case ListV(elements) => ListV(headVal :: elements)
        case _ => error("not a list")
    // tuple
    case ETuple(exprs: List[Expr]) => TupleV(exprs.map(interp(_, env)))
    // none
    case ENone => NoneV
    // some
    case ESome(value: Expr) => SomeV(interp(value, env))
    // let binding
    case ELet(pattern: Pattern, value: Expr, scope: Expr) =>
      val v = interp(value, env)
      matchPattern(pattern, v) match
        case Some(bindings) => interp(scope, env ++ bindings)
        case None => error("invalid pattern match")
    // mutually recursive function
    case ERec(funs: List[NamedFun], scope: Expr) =>
      lazy val recEnv: Env = funs.foldLeft(env){
        case (accEnv, NamedFun(name, param, body)) => accEnv + (name -> CloV(param, body, () => recEnv))
      }
      interp(scope, recEnv)
    // lambda function
    case EFun(param: Pattern, body: Expr) => CloV(param, body, () => env)

    // function application
    case EApp(fun: Expr, args: Expr) =>
      val funVal = interp(fun, env)
      val argVal = interp(args, env)
      funVal match
        case CloV(param, body, fEnv) =>
          matchPattern(param, argVal) match
            case Some(bindings) => interp(body, fEnv() ++ bindings)
            case None => error("not a function")
        case _ => error("not a function")
    // pattern matching
    case EMatch(value: Expr, cases: List[Case]) =>
      val v = interp(value, env)
      def matchCase(cases: List[Case]): Value = cases match
        case Nil => error("unmatched value")
        case Case(pattern, body) :: rest =>
          matchPattern(pattern, v) match
            case Some(bindings) => interp(body, env ++ bindings)
            case None => matchCase(rest)
      matchCase(cases)
  // ---------------------------------------------------------------------------
  // Problem #2
  // ---------------------------------------------------------------------------
  def hanoiMovesBody: String = """
    let rec concat l1 l2 = match l1 with
      | [] -> l2
      | h :: t -> h :: concat t l2
    and helper n source temp target =
      if n = 0 then []
      else concat (concat (helper (n - 1) source target temp) [(source, target)]) (helper (n - 1) temp source target)
    in helper n source temp target
  """
}
