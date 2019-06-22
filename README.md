# tfjs-backsolve
Back-substitution for rank2/1 tensors in tensorflow.js

Experimental. Use at own risk.

```js
const L = tfc.tensor2d([[2, 4, 8], [0, 16, 32], [0, 0, 64]])
const b = tfc.tensor1d([20, 400, 8000])
const result = await backsolve(L, b)
// [-40, -225, 125]
```
