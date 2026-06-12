package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*
  import Type.*

  def validType(ty: Type, tenv: TypeEnv): Unit = ty match
    case NumT | BoolT      => ()
    case ArrowT(ps, r)     => ps.foreach(validType(_, tenv)); validType(r, tenv)
    case NameT(name)       =>
      if (!tenv.tys.contains(name)) error(s"unknown type: $name")

  def typeCheck(expr: Expr, tenv: TypeEnv): Type = expr match
    case Num(_) => NumT
    case Bool(_) => BoolT
    case Add(l,r) => 
      val lt = typeCheck(l, tenv)
      val rt = typeCheck(r, tenv)
      (lt, rt) match
        case (NumT, NumT) => NumT
        case (l, r) => error(s"expected number but got ${lt.str} and ${rt.str}")
    case Mul(l,r) => 
      val lt = typeCheck(l, tenv)
      val rt = typeCheck(r, tenv)
      (lt, rt) match
        case (NumT, NumT) => NumT
        case (l, r) => error(s"expected number but got ${lt.str} and ${rt.str}")
    case Div(l,r) => 
      val lt = typeCheck(l, tenv)
      val rt = typeCheck(r, tenv)
      (lt, rt) match
        case (NumT, NumT) => NumT
        case (l, r) => error(s"expected number but got ${lt.str} and ${rt.str}")
    case Mod(l,r) => 
      val lt = typeCheck(l, tenv)
      val rt = typeCheck(r, tenv)
      (lt, rt) match
        case (NumT, NumT) => NumT
        case (l, r) => error(s"expected number but got ${lt.str} and ${rt.str}")
    case Eq(l,r) => 
      val lt = typeCheck(l, tenv)
      val rt = typeCheck(r, tenv)
      (lt, rt) match
        case (NumT, NumT) => BoolT
        case (BoolT, BoolT) => BoolT
        case (l, r) => error(s"expected number or boolean but got ${lt.str} and ${rt.str}")
    case Lt(l,r) =>
      val lt = typeCheck(l, tenv)
      val rt = typeCheck(r, tenv)
      (lt, rt) match
        case (NumT, NumT) => BoolT
        case (l, r) => error(s"expected number but got ${lt.str} and ${rt.str}")
    case Val(n, i, b) =>
      val valueTy = typeCheck(i, tenv)
      validType(valueTy, tenv)
      typeCheck(b, tenv.addVar(n -> valueTy))
    case Id(x) => tenv.vars.getOrElse(x, error(s"undefined variable: $x"))
    case Fun(params, body) =>
      for (param <- params) {
        validType(param.ty, tenv)
      }
      val paramTys = params.map(_.ty)
      val newTenv = tenv.addVars(params.map { p => p.name -> p.ty })
      val retTy = typeCheck(body, newTenv)
      validType(retTy, tenv)
      ArrowT(paramTys, retTy)
    case Rec(name, params, retTy, body, inExpr) =>
      for (param <- params) {
        validType(param.ty, tenv)
      }
      val paramTys = params.map(_.ty)
      val funType = ArrowT(paramTys, retTy)
      val newTenv = tenv
                      .addVar(name -> funType)
                      .addVars(params.map { p => p.name -> p.ty })
      val bodyRetType = typeCheck(body, newTenv)
      if (bodyRetType != retTy)
        error(
          s"function body has return type ${bodyRetType.str} but expected ${retTy.str}",
        )
      typeCheck(inExpr, tenv.addVar(name -> funType))
    case App(fun, args) =>
      typeCheck(fun, tenv) match
        case ArrowT(paramTys, retTy) =>
          if (paramTys.length != args.length)
            error(
              s"expected ${paramTys.length} arguments but got ${args.length}",
            )
          paramTys.zip(args).foreach { case (pt, arg) =>
            val at = typeCheck(arg, tenv)
            if (pt != at)
              error(
                s"expected argument of type ${pt.str} but got ${at.str}",
              )
          }
          retTy
        case t => error(s"expected function but got ${t.str}")
    case If(cond, thenP, elseP) =>
      if (typeCheck(cond, tenv) != BoolT)
        error(s"expected condition of type Boolean ")
      val thenTy = typeCheck(thenP, tenv)
      val elseTy = typeCheck(elseP, tenv)
      if (thenTy != elseTy)
        error(
          s"then and else branches must have the same type but got ${thenTy.str} and ${elseTy.str}",
        )
      thenTy
   
    case TypeDef(name, variants, body) =>
      if (tenv.tys.contains(name)) error(s"duplicate type definition: $name")
      val newTenv = tenv.addType(
                    name, 
                    variants.map (v => v.name -> v.ptys).toMap
                    )
      variants.foreach(v => 
          v.ptys.foreach(validType(_, newTenv)))
      
      val constructorTypes = variants.map { variant =>
        variant.name -> ArrowT(variant.ptys, NameT(name))
      }
      typeCheck(body, newTenv.addVars(constructorTypes))
    
    case Match(e, cs) => 
      val scrutineeTy = typeCheck(e, tenv)
      val tname = scrutineeTy match
        case NameT(n) => n
        case t => error(s"can only match on user-defined types but got ${t.str}")
      val variants = tenv.tys.getOrElse(tname, error(s"unknown type: $tname"))
      val caseNames = cs.map(_.name)
      if (caseNames.toSet.size != caseNames.size) error(s"duplicate case names in match: ${caseNames.mkString(", ")}")
      if (caseNames.toSet != variants.keySet) error(s"case names in match do not match variants of type $tname: expected ${variants.keySet.mkString(", ")} but got ${caseNames.mkString(", ")}")
      val tys = cs.map { c =>
        val vtys = variants.getOrElse(c.name, error(s"unknown case name: ${c.name}"))
        if (vtys.length != c.params.length) error(s"expected ${vtys.length} parameters for case ${c.name} but got ${c.params.length}")
        typeCheck(c.body, tenv.addVars(c.params.zip(vtys)))
      }
      tys.reduce((a, b) => if (a != b) error("branches differ") else a)
  
  
  
  def interp(expr: Expr, env: Env): Value = expr match
    case Num(n)       => NumV(n)
    case Bool(b)      => BoolV(b)
    case Add(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => NumV(n1 + n2)
      case _ => error(s"Runtime error: expected numbers in addition, got ${l.str} and ${r.str}")
    case Mul(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => NumV(n1 * n2)
      case _ => error(s"Runtime error: expected numbers in addition, got ${l.str} and ${r.str}")
    case Div(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => 
        if (n2 != 0) NumV(n1 / n2)
        else error(s"Runtime error: division by zero in ${l.str} / ${r.str}")
      case _ => error(s"Runtime error: expected numbers in division, got ${l.str} and ${r.str}")
    case Mod(l, r)    => (interp(l, env), interp(r, env)) match
      case (NumV(n1), NumV(n2)) => 
        if (n2 != 0) NumV(n1 % n2)
        else error(s"Runtime error: division by zero in ${l.str} / ${r.str}")
      case _ => error(s"Runtime error: expected numbers in division, got ${l.str} and ${r.str}")
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


    case Fun(ps, b)   => CloV(ps.map(_.name), b, () => env)


    case Rec(n, ps, rt, b, s) =>   

      lazy val recEnv: Env = env + (n -> CloV(ps.map(_.name), b, () => recEnv))
      interp(s, recEnv)

    case App(f, es)  => 
      val fv = interp(f, env)
      val argValues: List[Value] = es.map(arg => interp(arg,env))
      fv match
        case CloV(ps, b, cloEnv) =>
          interp(b, cloEnv()++ ps.zip(argValues) )
        
        case ConstrV(name) => VariantV(name, argValues)
        case _ => error(s"Runtime error: expected a function in application, got ${f.str}")

    case If(c, t, e) => 
      val cv = interp(c, env)
      cv match
        case BoolV(true) => interp(t, env)
        case BoolV(false) => interp(e, env)
        case _ => error(s"Runtime error: expected a boolean in if condition, got ${c.str}")

    case TypeDef(x, vs, b) =>
      val constructorEnv: Env = 
        vs.map{ v =>
          v.name -> ConstrV(v.name)
          }.toMap
      interp(b, env ++ constructorEnv)


    case Match(e, cs) => 
      interp(e, env) match
        case VariantV(vname, values) =>
          cs.find(_.name == vname) match
            case Some(MatchCase(_, params, body)) =>
              if (params.length != values.length)
                error(
                  s"Runtime error: expected ${params.length} fields " +
                    s"but got ${values.length}",
                )

              val caseEnv = env ++ params.zip(values)
              interp(body, caseEnv)
            case _ => error(
            s"Runtime error: no matching case for $vname",
              )
        case value =>
          error(
            s"Runtime error: expected a variant value in match " +
              s"but got ${value.str}",
          )
}
