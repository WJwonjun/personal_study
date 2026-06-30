package kuplrg

object Implementation extends Template {

  import Expr.*, RecDef.*, Value.*, Type.*, TypeInfo.*


  // ty가 tenv 내에서 유효한지 검사
  def mustwellFormed(ty:Type, tenv: TypeEnv): Unit =  ty match
    case NothingT | AnyT | UnitT | NumT | BoolT | StrT => ()
    case IdT(name, tys) => tenv.tys.get(name) match
      case Some(TIVar) => 
        if (tys.nonEmpty) error("type variable with args")
      case Some(TIAdt(tvars, variants)) =>
        if (tvars.length != tys.length) error("wrong type arg count")
        tys.foreach(ty => mustwellFormed(ty,tenv))
      case None => error("notwellformed") 
    case ArrowT(tvars, paramTys, retTy) =>
      val newtenv = tenv.addTypeVars(tvars)
      paramTys.foreach(paramty => mustwellFormed(paramty, newtenv))
      mustwellFormed(retTy, newtenv)
      



  def addDefToTenv(tenv: TypeEnv, rd: RecDef): TypeEnv = rd match
    case LazyVal(name, ty, init) => tenv.addVar(name -> ty)
    case RecFun(name, tvars, params, rty, body) => 
      val newty = ArrowT(tvars,params.map(_.ty), rty)
      tenv.addVar(name -> newty)
    case TypeDef(name, tvars:List[String], varts:List[Variant]) =>
      if (tenv.tys.contains(name)) error("duplicate type name")
      val inittenv = tenv.addTypeName(name, tvars, varts)
      varts.foldLeft(inittenv){(curTenv, v) => 
        val constrTy = ArrowT(tvars, v.params.map(_.ty), IdT(name, tvars.map(IdT(_))))
        curTenv.addVar(v.name -> constrTy)
       }
    // Variant(name: String, params: List[Param])
    // Param(name: String, ty: Type)

  def checkDef(tenv: TypeEnv, rd: RecDef): Unit = rd match
    case LazyVal(name,ty,init) =>
      mustwellFormed(ty, tenv)
      val retTy = typeCheck(init,tenv)
      if (!isSubtype(retTy, ty)) error("cd_lv")

    case RecFun(name, tvars, params, rty, body) => 
      tvars.foreach(tv =>
        if (tenv.tys.contains(tv)) error("duplicate type variable")
      )
      val newtenv = tenv.addTypeVars(tvars)
      params.foreach(p => mustwellFormed(p.ty, newtenv))
      val newtenv2 = newtenv.addVars(params.map(p=> p.name -> p.ty))
      val bodyty = typeCheck(body, newtenv2)
      if (!isSubtype(bodyty, rty)) error("rf3") 
      

    case TypeDef(name, tvars, varts) => 
      val newtenv = tenv.addTypeVars(tvars)
      varts.foreach(v => 
        v.params.foreach(p => mustwellFormed(p.ty, newtenv))  
        )

  def subst(ty: Type, map: Map[String, Type]): Type = ty match
    case IdT(name, tys) =>
      map.get(name) match
        case Some(replaced) => replaced              // 타입변수면 치환
        case None => IdT(name, tys.map(t => subst(t, map)))  // 아니면 인자만 재귀
    case ArrowT(tvars, pts, rty) =>
      // 주의: 안쪽에서 다시 묶이는(shadowing) 변수는 치환 제외
      val inner = map -- tvars
      ArrowT(tvars, pts.map(t => subst(t, inner)), subst(rty, inner))
    case _ => ty   // 기본 타입은 그대로
    
  def isSubtype(lty: Type, rty: Type): Boolean = (lty, rty) match
    case (NothingT,_) => true
    case (_, AnyT) => true
    case (IdT(lname, ltys), IdT(rname, rtys)) =>
      lname == rname &&
      ltys.length == rtys.length &&
      ltys.zip(rtys).forall((l,r) => isSubtype(l,r))
    case (ArrowT(ltvars, lpts, lrty), ArrowT(rtvars, rpts, rrty)) =>
      ltvars.length == rtvars.length && {
        // 오른쪽 변수 rtvars를 왼쪽 변수 ltvars로 갈아끼우는 맵
        val map = rtvars.zip(ltvars.map(a => IdT(a))).toMap
        val rpts2 = rpts.map(t => subst(t, map))   // 치환된 오른쪽 파라미터들
        val rrty2 = subst(rrty, map)               // 치환된 오른쪽 리턴
        lpts.length == rpts2.length &&
        lpts.zip(rpts2).forall((lp, rp) => isSubtype(rp, lp)) &&  // 반공변
        isSubtype(lrty, rrty2)                                     // 공변
      }
    case _ => lty == rty

  // lty, rty 합집합 type
  def join(lty:Type, rty:Type):Type =  
    if (isSubtype(lty,rty)) rty
    else if (isSubtype(rty,lty)) lty
    else
      (lty, rty) match
        case (IdT(ln, lts), IdT(rn, rts)) if ln == rn =>
          IdT(ln, lts.zip(rts).map((l, r) => join(l, r)))
        case (ArrowT(ltvars, lpts, lrty), ArrowT(rtvars, rpts, rrty))
            if ltvars.length == rtvars.length && lpts.length == rpts.length =>
          val map = rtvars.zip(ltvars.map(IdT(_))).toMap
          val rpts2 = rpts.map(t => subst(t, map))
          val rrty2 = subst(rrty, map)
          ArrowT(ltvars, lpts.zip(rpts2).map((lp, rp) => meet(lp, rp)), join(lrty, rrty2))
        case _ => AnyT

  // lty, rty 교집합 type  
  def meet(lty: Type, rty: Type): Type =
    if (isSubtype(lty, rty)) lty
    else if (isSubtype(rty, lty)) rty
    else (lty, rty) match
      case (IdT(ln, lts), IdT(rn, rts)) if ln == rn =>
        IdT(ln, lts.zip(rts).map((l, r) => meet(l, r)))
      case (ArrowT(ltvars, lpts, lrty), ArrowT(rtvars, rpts, rrty))
          if ltvars.length == rtvars.length && lpts.length == rpts.length =>
        val map = rtvars.zip(ltvars.map(IdT(_))).toMap
        val rpts2 = rpts.map(t => subst(t, map))
        val rrty2 = subst(rrty, map)
        ArrowT(ltvars, lpts.zip(rpts2).map((lp, rp) => join(lp, rp)), meet(lrty, rrty2))
      case _ => NothingT

  def typeCheck(expr: Expr, tenv: TypeEnv): Type = expr match
    case EUnit => UnitT
    // numbers
    case ENum(number: BigInt) => NumT
    // booleans
    case EBool(bool: Boolean) => BoolT
    // strings
    case EStr(string: String) => StrT
    // identifier lookups
    case EId(name: String) => tenv.vars.getOrElse(name, error("typecheck_Eid"))
    // addition
    case EAdd(left: Expr, right: Expr) => 
      val lty = typeCheck(left, tenv)
      val rty = typeCheck(right, tenv)
      if (isSubtype(lty, NumT) && isSubtype(rty, NumT)) NumT else error("eadd")
    // multiplication
    case EMul(left: Expr, right: Expr)  => 
      val lty = typeCheck(left, tenv)
      val rty = typeCheck(right, tenv)
      if (isSubtype(lty, NumT) && isSubtype(rty, NumT)) NumT else error("eadd")
    // division
    case EDiv(left: Expr, right: Expr)  => 
      val lty = typeCheck(left, tenv)
      val rty = typeCheck(right, tenv)
      if (isSubtype(lty, NumT) && isSubtype(rty, NumT)) NumT else error("eadd")
    // modulo
    case EMod(left: Expr, right: Expr)  => 
      val lty = typeCheck(left, tenv)
      val rty = typeCheck(right, tenv)
      if (isSubtype(lty, NumT) && isSubtype(rty, NumT)) NumT else error("eadd")
    // string concatenation
    case EConcat(left: Expr, right: Expr) => 
      val lty = typeCheck(left, tenv)
      val rty = typeCheck(right, tenv)
      if (isSubtype(lty, StrT) && isSubtype(rty, StrT)) StrT else error("eadd")
    // equal-to
    case EEq(left: Expr, right: Expr) =>
      val lty = typeCheck(left, tenv)
      val rty = typeCheck(right, tenv)
      BoolT
    // less-than
    case ELt(left: Expr, right: Expr) => 
      val lty = typeCheck(left, tenv)
      val rty = typeCheck(right, tenv)
      if (isSubtype(lty, NumT) && isSubtype(rty, NumT)) BoolT else error("eadd")
    // sequence
    case ESeq(left: Expr, right: Expr) => 
      val lty = typeCheck(left, tenv)
      typeCheck(right, tenv)

    // conditional
    case EIf(cond: Expr, thenExpr: Expr, elseExpr: Expr) => 
      val ty1 = typeCheck(cond, tenv)
      if (isSubtype(ty1, BoolT)) join(typeCheck(thenExpr,tenv),typeCheck(elseExpr,tenv)) 
      else error("tc_eif")
    
    // immutable variable definitions
    case EVal(x: String, tyOpt: Option[Type], expr: Expr, body: Expr) => tyOpt match
      case Some(ty0) =>
        val ty1 = typeCheck(expr, tenv)
        if (isSubtype(ty1,ty0)) typeCheck(body, tenv.addVar(x -> ty0)) else error("typecheck_eval")
      case None => 
        val ty1  =typeCheck(expr, tenv)
        val tenv1 = tenv.addVar(x -> ty1)
        typeCheck(body, tenv1)
    
    // anonymous (lambda) functions
    // Param(name: String, ty:Type)
    case EFun(params: List[Param], body: Expr) =>
      params.foreach(p=> mustwellFormed(p.ty,tenv))
      val newTenv = tenv.addVars(params.map(p=> p.name -> p.ty))
      val retTy = typeCheck(body, newTenv)
      ArrowT(Nil, params.map(_.ty),retTy)
    
    // function applications
    // tys: taus
    case EApp(fun: Expr, tys: List[Type], args: List[Expr]) => typeCheck(fun, tenv) match
      case ArrowT(tvars, paramTys, retTy) =>
        if (tvars.length != tys.length) error("eapp1")  
        tys.foreach(t => mustwellFormed(t,tenv))

        val map = tvars.zip(tys).toMap
        // 치환 맵
        // 예: tvars=["A","B"], tys=[NumT, StrT]
        //  →  Map("A" -> NumT, "B" -> StrT)

        val expectedparamtys = paramTys.map(pt => subst(pt, map))

        if (args.length != expectedparamtys.length) error("eapp2")

        args.zip(expectedparamtys).foreach{ (arg,expectedparamty) =>
          val argTy = typeCheck(arg, tenv)
          if(!isSubtype(argTy, expectedparamty)) error("eapp3")
        }
        subst(retTy, map)
      
      case _ => error("eapp4")


    // mutually recursive definitions
    case ERecDefs(defs, body) =>
      val finalTenv = defs.foldLeft(tenv) { (curTenv, d) =>
        addDefToTenv(curTenv, d)        // 오타 수정
      }
      defs.foreach(d => checkDef(finalTenv, d))   // 인자 순서: (tenv, rd)
      val bodyTy = typeCheck(body, finalTenv)
      mustwellFormed(bodyTy, tenv)
      bodyTy

    
    // pattern matching
    // case class MatchCase(name: String, params: List[String], body: Expr):
    case EMatch(expr: Expr, mcases: List[MatchCase]) => typeCheck(expr,tenv) match
      case IdT(name, typeArgs) =>
        tenv.tys.get(name) match
          case Some(TIAdt(tvars, variants)) =>
            val map = tvars.zip(typeArgs).toMap

            // case 이름들 (중복 포함)
            val caseNames = mcases.map(_.name)
            // ① 중복 검사: distinct하면 길이가 줄어듦
            if (caseNames.distinct.length != caseNames.length) error("duplicate case")
            // ② exhaustive 검사: case 이름 집합 == 변이 이름 집합
            if (caseNames.toSet != variants.keySet) error("non-exhaustive or unknown case")

            val caseTypes = mcases.map { mcase =>
              val fieldParams = variants.getOrElse(mcase.name, error("no such variant"))
              if (mcase.params.length != fieldParams.length) error("wrong pattern var count")
              val fieldTypes = fieldParams.map(p => subst(p.ty, map))
              val newtenv = tenv.addVars(mcase.params.zip(fieldTypes))
              typeCheck(mcase.body, newtenv)
            }
            caseTypes.reduce((a, b) => join(a, b))
          case _ => error("not an act")
      case _ => error("not enum")

    case EExit(expr: Expr) => 
      typeCheck(expr, tenv)
      NothingT

  def eq(lv:Value, rv:Value): Boolean = (lv,rv) match 
    case (UnitV, UnitV) => true
    case (NumV(l),NumV(r)) => l==r
    case (BoolV(l),BoolV(r)) => l==r
    case (StrV(l),StrV(r)) => l==r
    case (VariantV(ln,lvs),VariantV(rn,rvs)) => 
      ln == rn &&
      lvs.length == rvs.length &&
      lvs.zip(rvs).forall((l,r) => eq(l,r)) 
    case _ => false
  
  def addDefToEnv(env: Env, d: RecDef, finalEnv: () => Env): Env = d match
    case LazyVal(name, ty, init) =>
      env + (name -> ExprV(init, finalEnv))           // ExprV로 (lazy)
    case RecFun(name, tvars, params, rty, body) =>
      env + (name -> CloV(params.map(_.name), body, finalEnv))  // CloV
    case TypeDef(name, tvars, varts) =>
      varts.foldLeft(env) { (e, v) =>
        e + (v.name -> ConstrV(v.name))               // 각 생성자 ConstrV
      }

  def interp(expr: Expr, env: Env): Value = expr match
    case EUnit => UnitV
    case ENum(number: BigInt) => NumV(number)
    // booleans
    case EBool(bool: Boolean) => BoolV(bool)
    // strings
    case EStr(string: String) => StrV(string)
    // identifier lookups
    case EId(name) => env.getOrElse(name, error("interp_id")) match
      case ExprV(e, lenv) => interp(e, lenv())   // lazy val이면 지금 평가!
      case v => v  
    case EAdd(left: Expr, right: Expr) => (interp(left, env),interp(right, env)) match
      case (NumV(l), NumV(r)) => NumV(l+r)
      case _ => error("eadd")
    // multiplication
    case EMul(left: Expr, right: Expr) => (interp(left, env),interp(right, env)) match
      case (NumV(l), NumV(r)) => NumV(l*r)
      case _ => error("emul")
    // division
    case EDiv(left: Expr, right: Expr) => (interp(left, env),interp(right, env)) match
      case (NumV(l), NumV(0)) => error("zero")
      case (NumV(l), NumV(r)) => NumV(l/r)
      case _ => error("emul")
    // modulo
    case EMod(left: Expr, right: Expr) => (interp(left, env),interp(right, env)) match
      case (NumV(l), NumV(0)) => error("zero")
      case (NumV(l), NumV(r)) => NumV(l%r)
      case _ => error("emul")
    // string concatenation
    case EConcat(left: Expr, right: Expr) =>(interp(left, env),interp(right, env)) match
      case (StrV(l), StrV(r)) => StrV(l++r)
      case _ => error("econcat")
    // equal-to
    case EEq(left: Expr, right: Expr) => BoolV(eq(interp(left,env),interp(right,env)))
    // less-than
    case ELt(left, right) => (interp(left, env), interp(right, env)) match
      case (NumV(l), NumV(r)) => BoolV(l < r)
      case _ => error("elt")
    // sequence
    case ESeq(left: Expr, right: Expr) =>
      interp(left, env)
      interp(right, env)
    // conditional
    case EIf(cond: Expr, thenExpr: Expr, elseExpr: Expr) => interp(cond,env) match
      case BoolV(true) => interp(thenExpr,env)
      case BoolV(false) => interp(elseExpr,env)
      case _ => error("no")
    // immutable variable definitions
    case EVal(x: String, tyOpt: Option[Type], expr: Expr, body: Expr) => 
      val exprv = interp(expr, env)
      interp(body, env + (x -> exprv))
    // anonymous (lambda) functions
    // case class Param(name: String, ty: Type):
    case EFun(params: List[Param], body: Expr) =>
      CloV(params.map(_.name),body,() => env)
    // function applications
    case EApp(fun: Expr, tys: List[Type], args: List[Expr]) => interp(fun, env) match
      case CloV(params, body, fenv) => 
        val argVs = args.map(a => interp(a, env)) 
        val newenv = fenv() ++ params.zip(argVs).toMap
        interp(body, newenv)
      case ConstrV(name) =>
        val argVs = args.map(a => interp(a,env))
        VariantV(name,argVs)
      case _ => error("eapp")
    // mutually recursive definitions
    case ERecDefs(defs: List[RecDef], body: Expr) =>
      lazy val finalEnv: Env = defs.foldLeft(env){ (curEnv, d) =>
        addDefToEnv(curEnv,d,() => finalEnv)
      }
      interp(body, finalEnv)
    // pattern matching
    // case class MatchCase(name: String, params: List[String], body: Expr)
    case EMatch(expr, mcases) => interp(expr, env) match
      case VariantV(ename, values) =>
        mcases.find(mcase => mcase.name == ename) match
          case Some(mcase) =>
            val newmap = mcase.params.zip(values).toMap
            interp(mcase.body, env ++ newmap)
          case None => error("no matching case")
      case _ => error("match on non-variant")

    case EExit(expr: Expr) => error("program exit")
}
