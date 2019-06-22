"use strict"

const test = require("tape")
const backsolve = require(".")
const tfc = require("@tensorflow/tfjs-core")

const makeArrayEqual = (t) => (a, b) => {
  for (let i = 0; i < a.length; i++) {
    t.equal(a[i], b[i])
  }
}

test("solve example #1", async (t) => {
  const L = tfc.tensor2d([[2, 4, 8], [0, 16, 32], [0, 0, 64]])
  const b = tfc.tensor1d([20, 400, 8000])
  const expected = tfc.tensor1d([-40, -225, 125], "float32").arraySync()
  const result = (await backsolve(L, b)).arraySync()
  const arrayEqual = makeArrayEqual(t)
  arrayEqual(result, expected)
  t.end()
})