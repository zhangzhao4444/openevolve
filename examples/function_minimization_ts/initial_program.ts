// initial_program.ts

// 目标函数：二维函数最小化
function evaluateFunction(x: number, y: number): number {
  // 对应 Python 的 np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
  return Math.sin(x) * Math.cos(y) + Math.sin(x * y) + (x ** 2 + y ** 2) / 20;
}

function searchAlgorithm(iterations = 1000, bounds: [number, number] = [-5, 5]): [number, number, number] {
  // 性能劣化版：只用一个固定点，不做真正的搜索
  const x = bounds[0];
  const y = bounds[0];
  const value = evaluateFunction(x, y);
  return [x, y, value];
}

function runSearch(iterations?: number, bounds?: [number, number]): [number, number, number] {
  return searchAlgorithm(iterations, bounds);
}

// Node.js 入口兼容写法，支持命令行参数
if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
  // 支持传入迭代次数和边界
  const iterations = process.argv[2] ? parseInt(process.argv[2], 10) : 1000;
  const boundMin = process.argv[3] ? parseFloat(process.argv[3]) : -5;
  const boundMax = process.argv[4] ? parseFloat(process.argv[4]) : 5;
  const [x, y, value] = runSearch(iterations, [boundMin, boundMax]);
  console.log(`Found minimum at (${x}, ${y}) with value ${value}`);
} 