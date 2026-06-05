package kuplrg

object Implementation extends Template {

  import Stmt.*, Expr.*, Value.*, BOp.*, Inst.*, Control.*, Error.*

  def reduce(st: State): State =
    val State(k, s, h, m) = st
    def typeError: State = State(IRaise(TypeError) :: Nil, s, h, m)

    def fresh(mem: Mem): Addr =
      if (mem.isEmpty) 0 else mem.keys.max + 1

    def alloc(mem: Mem, value: Value): (Addr, Mem) =
      val addr = fresh(mem)
      (addr, mem + (addr -> value))

    def localVars(block: Block): Set[String] =
      block.stmts.flatMap {
        case SAssign(x, _)      => Set(x)
        case SDef(x, _, _)      => Set(x)
        case SIf(_, tb, eb)     => localVars(tb) ++ localVars(eb)
        case SWhile(_, body)    => localVars(body)
        case STry(body, except) => localVars(body) ++ localVars(except)
        case _                  => Set.empty
      }.toSet

    def hasYield(block: Block): Boolean =
      block.stmts.exists {
        case SYield(_)          => true
        case SDef(_, _, body)   => hasYield(body)
        case SIf(_, tb, eb)     => hasYield(tb) || hasYield(eb)
        case SWhile(_, body)    => hasYield(body)
        case STry(body, except) => hasYield(body) || hasYield(except)
        case _                  => false
      }

    def isSame(v0: Value, v1: Value): Boolean = (v0, v1) match
      case (NoneV, NoneV)         => true
      case (NumV(n0), NumV(n1))   => n0 == n1
      case (BoolV(b0), BoolV(b1)) => b0 == b1
      case (AddrV(a0), AddrV(a1)) => a0 == a1
      case _                      => false

    def equal(v0: Value, v1: Value, mem: Mem): Boolean =
      if (isSame(v0, v1)) true
      else (v0, v1) match
        case (AddrV(a0), AddrV(a1)) =>
          mem.get(a0).zip(mem.get(a1)).exists((x, y) => equal(x, y, mem))
        case (ListV(xs), ListV(ys)) =>
          xs.length == ys.length && xs.zip(ys).forall((x, y) => equal(x, y, mem))
        case _ => false

    def lessThan(v0: Value, v1: Value, mem: Mem): Option[Boolean] = (v0, v1) match
      case (NumV(n0), NumV(n1)) => Some(n0 < n1)
      case (AddrV(a0), AddrV(a1)) =>
        mem.get(a0).zip(mem.get(a1)).flatMap((x, y) => lessThan(x, y, mem))
      case (ListV(Nil), ListV(Nil)) => Some(false)
      case (ListV(Nil), ListV(_))   => Some(true)
      case (ListV(_), ListV(Nil))   => Some(false)
      case (ListV(x :: xs), ListV(y :: ys)) =>
        lessThan(x, y, mem).flatMap {
          case true => Some(true)
          case false =>
            if (!equal(x, y, mem)) Some(false)
            else lessThan(ListV(xs), ListV(ys), mem)
        }
      case _ => None

    def truthy(v: Value, mem: Mem): Boolean = v match
      case NoneV     => false
      case NumV(n)   => n != 0
      case BoolV(b)  => b
      case AddrV(a)  => mem.get(a).exists(truthy(_, mem))
      case ListV(xs) => xs.nonEmpty
      case _         => true

    def indexOf(n: BigInt, size: Int): Option[Int] =
      val idx = if (n < 0) BigInt(size) + n else n
      if (0 <= idx && idx < size && idx.isValidInt) Some(idx.toInt) else None

    def callValue(
      addr: Addr,
      args: List[Value],
      rest: Stack,
      k1: Cont,
      h1: Handler,
      mem: Mem,
    ): State =
      mem.get(addr) match
        case Some(CloV(params, body, fenv)) if params.length == args.length =>
          val (paramEnv, mem1) = params.zip(args).foldLeft((fenv, mem)) {
            case ((envAcc, memAcc), (x, v)) =>
              val (a, nextMem) = alloc(memAcc, v)
              (envAcc + (x -> a), nextMem)
          }
          val (bodyEnv, mem2) = (localVars(body) -- params).foldLeft((paramEnv, mem1)) {
            case ((envAcc, memAcc), x) =>
              val (a, nextMem) = alloc(memAcc, NoneV)
              (envAcc + (x -> a), nextMem)
          }
          val bodyHandler =
            (h1 + (Return -> KValue(k1, rest, h1))) - Break - Continue - Yield
          State(IBlock(bodyEnv, body) :: IReturn :: Nil, NoneV :: Nil, bodyHandler, mem2)

        case Some(GenV(params, body, fenv)) if params.length == args.length =>
          val (paramEnv, mem1) = params.zip(args).foldLeft((fenv, mem)) {
            case ((envAcc, memAcc), (x, v)) =>
              val (a, nextMem) = alloc(memAcc, v)
              (envAcc + (x -> a), nextMem)
          }
          val (bodyEnv, mem2) = (localVars(body) -- params).foldLeft((paramEnv, mem1)) {
            case ((envAcc, memAcc), x) =>
              val (a, nextMem) = alloc(memAcc, NoneV)
              (envAcc + (x -> a), nextMem)
          }
          val bodyHandler =
            (h1 + (Return -> KValue(k1, rest, h1))) - Break - Continue - Yield
          val nextK = IBlock(bodyEnv, body) :: IReturn :: Nil
          val (contAddr, mem3) = alloc(mem2, ContV(KValue(nextK, NoneV :: Nil, bodyHandler)))
          val (iterAddr, mem4) = alloc(mem3, IterV(contAddr, 0))
          State(k1, AddrV(iterAddr) :: rest, h1, mem4)

        case _ => State(IRaise(TypeError) :: Nil, rest, h1, mem)

    (k, s) match
      case (IStmt(_, SPass) :: k1, _) => State(k1, s, h, m)
      case (IStmt(env, SExpr(e)) :: k1, _) =>
        State(IExpr(env, e) :: IDrop :: k1, s, h, m)
      case (IStmt(env, SAssign(x, e)) :: k1, _) =>
        env.get(x).fold(State(IRaise(NameError(x)) :: Nil, s, h, m))(addr =>
          State(IExpr(env, e) :: IWrite(addr) :: k1, s, h, m),
        )
      case (IStmt(env, SSetItem(base, idx, e)) :: k1, _) =>
        State(IExpr(env, e) :: IExpr(env, base) :: IExpr(env, idx) :: ISetItem :: k1, s, h, m)
      case (IStmt(env, SIf(cond, thenBlock, elseBlock)) :: k1, _) =>
        val thenK = KValue(IBlock(env, thenBlock) :: k1, s, h)
        State(IExpr(env, cond) :: IJmpIf(thenK) :: IBlock(env, elseBlock) :: k1, s, h, m)
      case (IStmt(env, SWhile(cond, body)) :: k1, _) =>
        val continueK = KValue(IStmt(env, SWhile(cond, body)) :: k1, s, h)
        val breakK = KValue(k1, s, h)
        val bodyHandler = h + (Continue -> continueK) + (Break -> breakK)
        val bodyK = KValue(IBlock(env, body) :: IStmt(env, SWhile(cond, body)) :: k1, s, bodyHandler)
        State(IExpr(env, cond) :: IJmpIf(bodyK) :: k1, s, h, m)
      case (IStmt(_, SBreak) :: k1, _) =>
        State(IJmp(Break) :: k1, s, h, m)
      case (IStmt(_, SContinue) :: k1, _) =>
        State(IJmp(Continue) :: k1, s, h, m)
      case (IStmt(env, STry(body, except)) :: k1, _) =>
        val raiseK = KValue(IBlock(env, except) :: k1, s, h)
        val finallyK = KValue(k1, s, h)
        val bodyHandler = h + (Raise -> raiseK) + (Finally -> finallyK)
        State(IBlock(env, body) :: IJmp(Finally) :: Nil, s, bodyHandler, m)
      case (IStmt(_, SRaise) :: _, _) =>
        State(IRaise(RuntimeError) :: Nil, s, h, m)
      case (IStmt(env, SDef(x, params, body)) :: k1, _) =>
        env.get(x).fold(State(IRaise(NameError(x)) :: Nil, s, h, m)) { target =>
          val value =
            if (hasYield(body)) GenV(params, body, env)
            else CloV(params, body, env)
          val (addr, mem1) = alloc(m, value)
          State(IWrite(target) :: k1, AddrV(addr) :: s, h, mem1)
        }
      case (IStmt(env, SReturn(e)) :: k1, _) =>
        State(IExpr(env, e) :: IReturn :: k1, s, h, m)
      case (IStmt(env, SYield(e)) :: k1, _) =>
        State(IExpr(env, e) :: IYield :: k1, s, h, m)

      case (IBlock(env, Block(stmts)) :: k1, _) =>
        State(stmts.map(IStmt(env, _)) ::: k1, s, h, m)

      case (IExpr(_, ENone) :: k1, _) => State(k1, NoneV :: s, h, m)
      case (IExpr(_, ENum(n)) :: k1, _) => State(k1, NumV(n) :: s, h, m)
      case (IExpr(_, EBool(b)) :: k1, _) => State(k1, BoolV(b) :: s, h, m)
      case (IExpr(env, EId(x)) :: k1, _) =>
        env.get(x) match
          case Some(addr) => State(k1, m.getOrElse(addr, NoneV) :: s, h, m)
          case None       => State(IRaise(NameError(x)) :: Nil, s, h, m)
      case (IExpr(env, EBOp(op, left, right)) :: k1, _) =>
        State(IExpr(env, left) :: IExpr(env, right) :: IBOp(op) :: k1, s, h, m)
      case (IExpr(env, EList(elements)) :: k1, _) =>
        State(elements.map(IExpr(env, _)) ::: IList(elements.length) :: k1, s, h, m)
      case (IExpr(env, EAppend(list, elem)) :: k1, _) =>
        State(IExpr(env, list) :: IExpr(env, elem) :: IAppend :: k1, s, h, m)
      case (IExpr(env, EGetItem(list, idx)) :: k1, _) =>
        State(IExpr(env, list) :: IExpr(env, idx) :: IGetItem :: k1, s, h, m)
      case (IExpr(env, ELambda(params, body)) :: k1, _) =>
        val (addr, mem1) = alloc(m, CloV(params, Block(SReturn(body)), env))
        State(k1, AddrV(addr) :: s, h, mem1)
      case (IExpr(env, EApp(fun, args)) :: k1, _) =>
        State(IExpr(env, fun) :: args.map(IExpr(env, _)) ::: ICall(args.length) :: k1, s, h, m)
      case (IExpr(env, ECond(cond, thenExpr, elseExpr)) :: k1, _) =>
        val thenK = KValue(IExpr(env, thenExpr) :: k1, s, h)
        State(IExpr(env, cond) :: IJmpIf(thenK) :: IExpr(env, elseExpr) :: k1, s, h, m)
      case (IExpr(env, EIter(e)) :: k1, _) =>
        State(IExpr(env, e) :: IIter :: k1, s, h, m)
      case (IExpr(env, ENext(e)) :: k1, _) =>
        State(IExpr(env, e) :: INext :: k1, s, h, m)

      case (IBOp(Add) :: k1, NumV(n2) :: NumV(n1) :: s1) =>
        State(k1, NumV(n1 + n2) :: s1, h, m)
      case (IBOp(Mul) :: k1, NumV(n2) :: NumV(n1) :: s1) =>
        State(k1, NumV(n1 * n2) :: s1, h, m)
      case (IBOp(Div) :: _, NumV(0) :: NumV(_) :: s1) =>
        State(IRaise(ZeroDivisionError) :: Nil, s1, h, m)
      case (IBOp(Div) :: k1, NumV(n2) :: NumV(n1) :: s1) =>
        State(k1, NumV(n1 / n2) :: s1, h, m)
      case (IBOp(Mod) :: _, NumV(0) :: NumV(_) :: s1) =>
        State(IRaise(ZeroDivisionError) :: Nil, s1, h, m)
      case (IBOp(Mod) :: k1, NumV(n2) :: NumV(n1) :: s1) =>
        State(k1, NumV(n1 % n2) :: s1, h, m)
      case (IBOp(Eq) :: k1, v2 :: v1 :: s1) =>
        State(k1, BoolV(equal(v1, v2, m)) :: s1, h, m)
      case (IBOp(Is) :: k1, v2 :: v1 :: s1) =>
        State(k1, BoolV(isSame(v1, v2)) :: s1, h, m)
      case (IBOp(Lt) :: k1, v2 :: v1 :: s1) =>
        lessThan(v1, v2, m).fold(State(IRaise(TypeError) :: Nil, s1, h, m))(b =>
          State(k1, BoolV(b) :: s1, h, m),
        )
      case (IBOp(Lte) :: k1, v2 :: v1 :: s1) =>
        lessThan(v1, v2, m).fold(State(IRaise(TypeError) :: Nil, s1, h, m))(b =>
          State(k1, BoolV(b || equal(v1, v2, m)) :: s1, h, m),
        )

      case (IWrite(addr) :: k1, v :: s1) =>
        State(k1, s1, h, m + (addr -> v))
      case (IGetItem :: k1, NumV(n) :: AddrV(addr) :: s1) =>
        m.get(addr) match
          case Some(ListV(elements)) =>
            indexOf(n, elements.length).fold(State(IRaise(IndexError) :: Nil, s1, h, m))(idx =>
              State(k1, elements(idx) :: s1, h, m),
            )
          case _ => State(IRaise(TypeError) :: Nil, s1, h, m)
      case (ISetItem :: k1, NumV(n) :: AddrV(addr) :: v :: s1) =>
        m.get(addr) match
          case Some(ListV(elements)) =>
            indexOf(n, elements.length).fold(State(IRaise(IndexError) :: Nil, s1, h, m)) { idx =>
              State(k1, s1, h, m + (addr -> ListV(elements.updated(idx, v))))
            }
          case _ => State(IRaise(TypeError) :: Nil, s1, h, m)
      case (IList(n) :: k1, stack) if stack.length >= n =>
        val values = stack.take(n).reverse
        val rest = stack.drop(n)
        val (addr, mem1) = alloc(m, ListV(values))
        State(k1, AddrV(addr) :: rest, h, mem1)
      case (IAppend :: k1, v :: AddrV(addr) :: s1) =>
        m.get(addr) match
          case Some(ListV(elements)) =>
            State(k1, AddrV(addr) :: s1, h, m + (addr -> ListV(elements :+ v)))
          case _ => State(IRaise(TypeError) :: Nil, s1, h, m)
      case (IJmpIf(kv) :: k1, v :: s1) =>
        if (truthy(v, m)) State(kv.cont, kv.stack, kv.handler, m)
        else State(k1, s1, h, m)
      case (IJmp(control) :: _, _) =>
        h.get(control).fold(error(s"missing handler: ${control.str}")) { kv =>
          State(kv.cont, kv.stack, kv.handler, m)
        }
      case (IRaise(err) :: _, _) =>
        if (h.contains(Raise)) State(IJmp(Raise) :: Nil, s, h, m)
        else error(err.str)
      case (ICall(n) :: k1, stack) if stack.length >= n + 1 =>
        val rawArgs = stack.take(n)
        stack.drop(n) match
          case AddrV(addr) :: rest => callValue(addr, rawArgs.reverse, rest, k1, h, m)
          case _                   => State(IRaise(TypeError) :: Nil, stack.drop(n), h, m)
      case (IReturn :: _, v :: _) =>
        h.get(Return).fold(State(IRaise(TypeError) :: Nil, s, h, m)) { kv =>
          State(kv.cont, v :: kv.stack, kv.handler, m)
        }
      case (IYield :: k1, v :: s1) =>
        h.get(Yield).fold(State(IRaise(TypeError) :: Nil, s1, h, m)) { kv =>
          State(kv.cont, ContV(KValue(k1, s1, h)) :: v :: kv.stack, kv.handler, m)
        }
      case (IIter :: k1, AddrV(addr) :: s1) =>
        m.get(addr) match
          case Some(IterV(_, _)) => State(k1, AddrV(addr) :: s1, h, m)
          case Some(ListV(_)) =>
            val (iterAddr, mem1) = alloc(m, IterV(addr, 0))
            State(k1, AddrV(iterAddr) :: s1, h, mem1)
          case _ => State(IRaise(TypeError) :: Nil, s1, h, m)
      case (INext :: k1, AddrV(addr) :: s1) =>
        m.get(addr) match
          case Some(IterV(target, idx)) =>
            m.get(target) match
              case Some(ContV(kv)) =>
                val yieldK = KValue(IWrite(target) :: k1, s1, h)
                val returnK = KValue(IDrop :: IRaise(StopIteration) :: Nil, s1, h)
                val nextHandler = kv.handler + (Yield -> yieldK) + (Return -> returnK)
                State(kv.cont, kv.stack, nextHandler, m)
              case Some(ListV(elements)) =>
                if (idx < elements.length)
                  State(k1, elements(idx) :: s1, h, m + (addr -> IterV(target, idx + 1)))
                else State(IRaise(StopIteration) :: Nil, s1, h, m)
              case _ => State(IRaise(TypeError) :: Nil, s1, h, m)
          case _ => State(IRaise(TypeError) :: Nil, s1, h, m)
      case (IDrop :: k1, _ :: s1) =>
        State(k1, s1, h, m)

      case _ => typeError

  def locals(block: Block): Set[String] = 
    block.stmts.flatMap {
      case SAssign(x, _)      => Set(x)
      case SDef(x, _, _)      => Set(x)
      case SIf(_, tb, eb)     => locals(tb) ++ locals(eb)
      case SWhile(_, body)    => locals(body)
      case STry(body, except) => locals(body) ++ locals(except)
      case _                  => Set.empty
    }.toSet
}
