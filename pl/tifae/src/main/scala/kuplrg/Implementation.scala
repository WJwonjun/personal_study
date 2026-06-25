package kuplrg

object Implementation extends Template {

  import Expr.*
  import Value.*
  import Type.*

  // k가 bty에 포함되어 있으면 true, 없으면 false
  def occurs(k: Int, ty: Type): Boolean = ty match
    case VarT(l) => k == l
    case ArrowT(pty, rty) => occurs(k, pty) || occurs(k, rty)
    case _ => false
  
  def isSame(lty: Type, rty: Type): Boolean = (lty, rty) match
    case (NumT, NumT) => true
    case (BoolT, BoolT) => true
    case (ArrowT(lp,lr),ArrowT(rp,rr)) => isSame(lp,rp) && isSame(lr,rr)
    case (VarT(l),VarT(r)) => l==r
    case _ => false

  def newTypeVar(sol: Solution): (Type, Solution) =
    val k = sol.keys.maxOption.getOrElse(-1) + 1
    (VarT(k), sol + (k -> None))
  // Solution에 q를 a로 집어넣기
  def unify(lty:Type, rty:Type, sol: Solution) : Solution =
    (resolve(lty, sol), resolve(rty, sol)) match
      case (NumT, NumT) => sol
      case (BoolT, BoolT) => sol
      case (ArrowT(lpty, lrty), ArrowT(rpty, rrty)) => unify(lrty, rrty, unify(lpty, rpty, sol))
      case (VarT(k), VarT(l)) if k == l => sol
      case (VarT(k), rty) if !occurs(k, rty) => sol + (k -> Some(rty))
      case (lty, VarT(k)) if !occurs(k, lty) => sol + (k -> Some(lty))
      case _ => error(s"Cannot unify ${lty.str} and ${rty.str}")
  
  // Type -> TypeScheme
  def gen(ty: Type, env: TypeEnv, sol: Solution): TypeScheme =
    val ks = (free(ty, sol) -- freeEnv(env, sol)).toList
    TypeScheme(ks, ty)
  // TypeScheme -> Type
  // 타입 스킴 → 일반 타입 (양화 변수를 새 변수로)

  def inst(ts: TypeScheme, sol: Solution): (Type, Solution) =
    val base = sol.keys.maxOption.getOrElse(-1) + 1
    val mapping = ts.ks.zipWithIndex.map { case (old, i) => old -> (base + i) }.toMap

    // fresh를 먼저 None으로 등록 (이게 newSol이자 base)
    val newSol = mapping.values.foldLeft(sol) { (s, fresh) => s + (fresh -> None) }

    // tempSol: newSol 위에 old -> VarT(fresh) 매핑 추가
    val tempSol = mapping.foldLeft(newSol) {
      case (s, (old, fresh)) => s + (old -> Some(VarT(fresh)))
    }

    (resolve(ts.ty, tempSol), newSol)



  // ty의 체인 따라가서 진짜 타입 찾아줌
  def resolve(ty: Type, sol: Solution): Type = ty match
    case VarT(k) => sol(k) match
      case Some(t) => resolve(t, sol)   // 풀려 있으면 계속 따라감
      case None    => ty                // 아직 •이면 멈춤
    case ArrowT(pty, rty) => ArrowT(resolve(pty, sol),resolve(rty, sol))

    case _ => ty

  // solution에서 아직 정답 못 찾은 변수들 모임
  def free(ty: Type, sol: Solution): Set[Int] = ty match
    case VarT(k) => sol(k) match
      case None      => Set(k)              // 아직 • → 자유 변수
      case Some(ty2) => free(ty2, sol)     // 풀렸으면 따라 들어감
    case ArrowT(p, r) => free(p, sol) ++ free(r, sol)
    case _ => Set()                          // num, bool → 없음

  def freeEnv(env: TypeEnv, sol: Solution): Set[Int] =
    env.values.flatMap(ts => freeScheme(ts, sol)).toSet

  def freeScheme(ts: TypeScheme, sol: Solution): Set[Int] =
    free(ts.ty, sol) -- ts.ks.toSet
        
  def typeCheck(
    expr: Expr,
    tenv: TypeEnv,
    sol: Solution,
  ): (Type, Solution) = expr match
    case Num(num: BigInt) => (NumT, sol)
    // booleans
    case Bool(bool: Boolean) => (BoolT, sol)
    // addition
    case Add(left: Expr, right: Expr) => 
      val (leftty, leftsol) = typeCheck(left, tenv, sol)
      val (rightty, rightsol) = typeCheck(right, tenv, leftsol)
      val newsol = unify(leftty, NumT,rightsol)
      (NumT, unify(rightty, NumT, newsol))
    // multiplication
    case Mul(left, right) =>
      val (leftty, leftsol) = typeCheck(left, tenv, sol)
      val (rightty, rightsol) = typeCheck(right, tenv, leftsol)   // sol → leftsol
      val newsol = unify(leftty, NumT, rightsol)
      (NumT, unify(rightty, NumT, newsol))

    case Div(left, right) =>
      val (leftty, leftsol) = typeCheck(left, tenv, sol)
      val (rightty, rightsol) = typeCheck(right, tenv, leftsol)
      val newsol = unify(leftty, NumT, rightsol)
      (NumT, unify(rightty, NumT, newsol))

    case Mod(left, right) =>
      val (leftty, leftsol) = typeCheck(left, tenv, sol)
      val (rightty, rightsol) = typeCheck(right, tenv, leftsol)
      val newsol = unify(leftty, NumT, rightsol)
      (NumT, unify(rightty, NumT, newsol))

    case Eq(left, right) =>
      val (leftty, leftsol) = typeCheck(left, tenv, sol)
      val (rightty, rightsol) = typeCheck(right, tenv, leftsol)
      val newsol = unify(leftty, NumT, rightsol)
      (BoolT, unify(rightty, NumT, newsol))

    case Lt(left, right) =>
      val (leftty, leftsol) = typeCheck(left, tenv, sol)
      val (rightty, rightsol) = typeCheck(right, tenv, leftsol)
      val newsol = unify(leftty, NumT, rightsol)
      (BoolT, unify(rightty, NumT, newsol))
    // immutable variable definition
    case Val(name: String, expr: Expr, body: Expr) =>
      val (ty1, sol1) = typeCheck(expr,tenv,sol)
      val tyscheme1 = gen(ty1, tenv, sol1)
      typeCheck(body, tenv + (name -> tyscheme1), sol1)
    // identifier lookups
    case Id(name: String) =>
      val tyscheme1 = tenv.getOrElse(name, error("invalid_id"))
      inst(tyscheme1, sol)
    // anonymous (lambda) functions
    case Fun(param: String, body: Expr) =>
      val (pty, sol1: Solution) = newTypeVar(sol)        // ← 새 타입 변수 발급!
      val (rty: Type, sol2: Solution) = typeCheck(body, tenv + (param -> TypeScheme(Nil, pty)), sol1)
      (ArrowT(pty, rty), sol2)
    // recursive functions
    case Rec(f, p, b, s) =>
      val (pty, sol1) = newTypeVar(sol)
      val (rty, sol2) = newTypeVar(sol1)
      val fty = ArrowT(pty, rty)
      val tenv1 = tenv + (f -> TypeScheme(Nil, fty))
      val tenv2 = tenv1 + (p -> TypeScheme(Nil, pty))
      val (bty: Type, sol3: Solution) = typeCheck(b, tenv2, sol2)
      val sol4 = unify(bty, rty, sol3)
      typeCheck(s, tenv1, sol4)
    // function applications
    case App(f, a) =>
      val (fty, sol1) = typeCheck(f, tenv, sol)
      val (aty, sol2) = typeCheck(a, tenv, sol1)
      val (rty, sol3) = newTypeVar(sol2)
      val sol4 = unify(ArrowT(aty, rty), fty, sol3)
      (rty, sol4)
    // conditional
    case If(c, t, e) =>
      val (cty, sol1) = typeCheck(c, tenv, sol)
      val (tty, sol2) = typeCheck(t, tenv, sol1)
      val (ety, sol3) = typeCheck(e, tenv, sol2)
      val sol4 = unify(cty, BoolT, sol3)
      val sol5 = unify(tty, ety, sol4)
      (tty, sol5)

  def interp(expr: Expr, env: Env): Value = expr match
    case Num(num: BigInt) =>NumV(num)
    // booleans
    case Bool(bool: Boolean) => BoolV(bool)
    // addition
    case Add(left: Expr, right: Expr) => (interp(left, env), interp(right, env)) match
      case (NumV(l),NumV(r)) => NumV(l+r)
      case (_,_) => error("interp_add")
    // multiplication
    case Mul(left: Expr, right: Expr) => (interp(left, env), interp(right, env)) match
      case (NumV(l),NumV(r)) => NumV(l*r)
      case (_,_) => error("interp_mul")
    // division
    case Div(left: Expr, right: Expr) => (interp(left, env), interp(right, env)) match
      case (NumV(l),NumV(0)) => error("zero division")
      case (NumV(l),NumV(r)) => NumV(l/r)
      case (_,_) => error("interp_div")
    // modulo
    case Mod(left: Expr, right: Expr) => (interp(left, env), interp(right, env)) match
      case (NumV(l),NumV(0)) => error("zero division")
      case (NumV(l),NumV(r)) => NumV(l%r)
      case (_,_) => error("interp_mod")
    // equal-to
    case Eq(left: Expr, right: Expr) => (interp(left, env), interp(right, env)) match
      case (NumV(l),NumV(r)) => if (l==r) BoolV(true) else BoolV(false)
      case (BoolV(l),BoolV(r)) => if (l==r) BoolV(true) else BoolV(false)
      case (_,_) => error("interp_eq")
    // less-than
    case Lt(left: Expr, right: Expr) => (interp(left, env), interp(right, env)) match
      case (NumV(l),NumV(r)) => if (l < r) BoolV(true) else BoolV(false)
      case (_,_) => error("interp_lt")
    // immutable variable definition
    case Val(name: String, expr: Expr, body: Expr) => 
      val newval = interp(expr,env)
      interp(body, env + (name -> newval))
    // identifier lookups
    case Id(name: String) => env.getOrElse(name, error("interp_id"))
    // anonymous (lambda) functions
    case Fun(param: String, body: Expr) => CloV(param, body, ( ) => env)
    // recursive functions
    case Rec(name, param, body, scope) =>
      lazy val newenv: Env = env + (name -> CloV(param, body, () => newenv))
      interp(scope, newenv)
    // function applications
    case App(fexpr, aexpr) => interp(fexpr, env) match
      case CloV(p, b, fenv) =>
        val v = interp(aexpr, env)
        interp(b, fenv() + (p -> v))     // b 실행 + fenv는 () 붙여 호출
      case _ => error("interp_app")
    // conditional
    case If(cond: Expr, texpr: Expr, eexpr: Expr) => interp(cond, env) match
      case(BoolV(true)) => interp(texpr, env)
      case(BoolV(false)) => interp(eexpr, env)
      case _ => error("interp_if")
}
