package kuplrg

object Implementation extends Template {

  // ---------------------------------------------------------------------------
  // Basic Data Types
  // ---------------------------------------------------------------------------
  def clamp(lower: Int, x: Int, upper: Int): Int = if (x>upper) upper else (if (x<lower) lower else x)

  def validName(name: String): Boolean = name.length>0 && name.length<=10 && name(0).isUpper

  // ---------------------------------------------------------------------------
  // Functions
  // ---------------------------------------------------------------------------
  def collatzLength(n: Int): Int = 
    if (n==1) 1
    else if (n%2==0) 1 + collatzLength(n/2)
    else 1 + collatzLength(3*n+1)

  def fixpoint(f: Int => Int): Int => Int = 
    n => 
    if (f(n)==n) n
    else fixpoint(f)(f(n))

  def applyK(f: Int => Int, k: Int): Int => Int = 
    n =>
    if (k==0) n
    else applyK(f,k-1)(f(n)) 

  // ---------------------------------------------------------------------------
  // Collections
  // ---------------------------------------------------------------------------
  def sumEven(l: List[Int]): Int = 
    l
      .filter(_%2==0)
      .foldLeft(0)(_ + _) 

  def double(l: List[Int]): List[Int] = 
    l.flatMap(x=> List(x,x))

  def generate(f: Int => Int): Int => List[Int] = 
    n =>
    if(f(n)==n) List(n)
    else n :: generate(f)(f(n))

  def join(l: Map[String, Int], r: Map[String, Int]): Map[String, Int] = 
    (l.keySet | r.keySet) // set(key)
    .map { k => k -> (l.getOrElse(k, 0) + r.getOrElse(k, 0))} //set(k,v)
    .toMap // map(k,v)

  def subsets(set: Set[Int]): List[Set[Int]] = 
    if (set.isEmpty) List()
    else{   
      val rest = subsets(set.tail)
      (rest ++ rest.map(_ + set.head) :+ Set(set.head)).sortWith((a, b) => a.toList.sorted.toString < b.toList.sorted.toString) 
    } 

  // ---------------------------------------------------------------------------
  // Trees
  // ---------------------------------------------------------------------------
  import Tree.*

  def heightOf(t: Tree): Int = t match
    case Leaf(n) => 0
    case Branch(l, n, r) => 1 + heightOf(l).max(heightOf(r))

  def max(t: Tree): Int = t match
    case Leaf(n) => n
    case Branch(l,n,r) => n.max(max(l)).max(max(r))

  def postorder(t: Tree): List[Int] = t match
    case Leaf(n) => List(n)
    case Branch(l,n,r) => postorder(l):::postorder(r):::List(n)

  def count(t: Tree, f: Int => Boolean): Int = t match
    case Leaf(n) => if (f(n)) 1 else 0
    case Branch(l,n,r) => count(l,f) + count(r,f) + (if (f(n)) 1 else 0) 

  def merge(left: Tree, right: Tree): Tree = (left, right) match
    case (Branch(ll, ln, lr), Branch(rl, rn, rr)) => Branch(merge(ll, rl), ln + rn, merge(lr, rr))
    case (Leaf(ln), Leaf(rn)) => Leaf(ln + rn)
    case (Leaf(ln), Branch(_, rn, _)) => Leaf(ln + rn)
    case (Branch(_, ln, _), Leaf(rn)) => Leaf(ln + rn)

  // ---------------------------------------------------------------------------
  // Boolean Expressions
  // ---------------------------------------------------------------------------
  import BE.*

  def isImply(expr: BE): Boolean = expr match
    case Imply(_,_) => true
    case _ => false
 
  def noAnd(expr: BE): Boolean = expr match
    case And(_,_) => false
    case Or(a,b) => noAnd(a)&&noAnd(b)
    case Imply(a,b) => noAnd(a)&&noAnd(b)
    case Not(a) => noAnd(a)
    case _ => true

  def subExprs(expr: BE): Set[BE] = expr match
    case Literal(_) => Set(expr)
    case Variable(_) => Set(expr)
    case And(a,b) => Set(expr)++subExprs(a)++subExprs(b)
    case Or(a,b) => Set(expr)++subExprs(a)++subExprs(b)
    case Imply(a,b) => Set(expr)++subExprs(a)++subExprs(b)
    case Not(a) => Set(expr)++subExprs(a)

  def getString(expr: BE): String = expr match
    case Literal(true) => "#t"
    case Literal(false) => "#f"
    case Variable(name) => name
    case And(left,right) => "(" + getString(left) + " && " + getString(right) + ")"
    case Or(left,right) => "(" + getString(left) + " || " + getString(right) + ")"
    case Imply(left,right) => "(" + getString(left) + " => " + getString(right) + ")"
    case Not(expr) => "!" + getString(expr)

  def eval(expr: BE, env: Map[String, Boolean]): Boolean = expr match
    case Literal(true) => true
    case Literal(false) => false
    case Variable(name) => env.getOrElse(name, false)
    case And(left,right) => eval(left,env) && eval(right,env)
    case Or(left,right) => eval(left,env) || eval(right,env)
    case Imply(left, right) => !eval(left, env) || eval(right, env) // imply => !left || right
    case Not(expr) => !(eval(expr,env))
}
