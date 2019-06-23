"use strict"

const tfc = require("@tensorflow/tfjs-core")

const backSolve = (L, b) => {
  return tfc.tidy(() => {
    const n = +b.shape[0]
    let tensorOut = tfc.tensor1d([])
    // TODO: make it a bit more readable and use a column vector throughout
    const lastB = b.gather([n - 1]).as1D()
    const lastL = L.gather([n - 1]).flatten().gather([n - 1]).as1D()
    tensorOut = tensorOut.concat(tfc.div(lastB, lastL).as1D())
    for (let i = n - 2; i >= 0; i--) {
      const Ls = tfc.slice(tfc.gather(L, i), i + 1, n - i - 1).reshape([1, n - i - 1])
      const t = tfc.matMul(Ls, tensorOut.reshape([n - i - 1, 1])).asScalar()
      const diag = tfc.gather(L, i).as1D().gather(i).asScalar()
      const bt = b.gather([i]).asScalar()
      const val = tfc.div(tfc.sub(bt, t), diag).as1D()
      tensorOut = val.concat(tensorOut)
    }
    return tensorOut.reshape([n, 1])
  })
}

module.exports = backSolve