package kuplrg

trait Template {

  import Type.*

  def eval(str: String): String =
    val expr = Expr(str)
    val t = toScheme(typeCheck(expr, Map.empty, Map.empty))
    val v = interp(expr, Map.empty)
    s"${v.str}: ${t.str}"

  // type checker
  def typeCheck(
    expr: Expr,
    tenv: TypeEnv,
    sol: Solution,
  ): (Type, Solution)

  // type resolution with solution
  def resolve(ty: Type, sol: Solution): Type

  // free type variables
  def free(ty: Type, sol: Solution): Set[Int]

  // interpreter
  def interp(expr: Expr, env: Env): Value

  // conversion from types to type schemes at the top level
  def toScheme(pair: (Type, Solution)): TypeScheme =
    val (ty, sol) = pair
    val ks = free(ty, sol).toList.sorted
    val ls = (1 to ks.length).toList
    val map = (ks zip ls).toMap
    def subst(ty: Type): Type = ty match
      case VarT(k)          => map.get(k).fold(ty)(VarT(_))
      case ArrowT(pty, rty) => ArrowT(subst(pty), subst(rty))
      case _                => ty
    TypeScheme(ls, subst(resolve(ty, sol)))
}
