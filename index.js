"use strict"

const tfc = require("@tensorflow/tfjs-core")

const backSolve = async (L, b) => {
  const n = +b.shape[0]
  const bufferOut = tfc.buffer([n], "float32")
  const buffer = await b.buffer()
  const Lbuffer = await L.buffer()

  bufferOut.set(buffer.get(n - 1) / Lbuffer.get(n - 1, n - 1), n - 1)
  for (let i = n - 2; i >= 0; i--) {
    const Ls = tfc.slice(tfc.gather(L, i, 1), i + 1, n - i - 1).reshape([1, n - i - 1])
    const bs = bufferOut.toTensor().slice(i + 1, n - i - 1).reshape([n - i - 1, 1])
    const t = tfc.matMul(Ls, bs).asScalar()
    const diag = tfc.gather(L, i).as1D().gather(i).asScalar()
    const bt = tfc.tensor(buffer.get(i), [])
    const val = tfc.div(tfc.sub(bt, t), diag)
    bufferOut.set((await val.buffer()).get(0), i)
    Ls.dispose()
    bs.dispose()
    t.dispose()
    diag.dispose()
    bt.dispose()
    val.dispose()
  }
  return bufferOut.toTensor()
}

module.exports = backSolve