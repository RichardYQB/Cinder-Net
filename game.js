// 游戏配置与状态
const C = { g: 20, t: 20, s: 150 }, // gridSize, tileCount, speed
    S = { // 游戏状态
        snake: [{ x: 10, y: 10 }],
        food: { x: 15, y: 15 },
        dx: 0, dy: 0, score: 0, speed: C.s,
        running: false, loop: null,
        high: localStorage.getItem('snakeHighScore') || 0
    },
    canvas = document.getElementById('gameCanvas'),
    ctx = canvas.getContext('2d'),
    E = id => document.getElementById(id);

canvas.width = canvas.height = C.t * C.g;

// 初始化
E('highScore').textContent = S.high;
[E('startBtn'), E('restartBtn')].forEach(b => b.onclick = startGame);
document.addEventListener('keydown', e => (keyMap[e.key] || keyMap[e.key.toLowerCase()])?.());

// 方向映射
const keyMap = {
    ArrowUp: () => move(0, -1), w: () => move(0, -1),
    ArrowDown: () => move(0, 1), s: () => move(0, 1),
    ArrowLeft: () => move(-1, 0), a: () => move(-1, 0),
    ArrowRight: () => move(1, 0), d: () => move(1, 0)
};

[['upBtn', 0, -1], ['downBtn', 0, 1], ['leftBtn', -1, 0], ['rightBtn', 1, 0]]
    .forEach(([id, dx, dy]) => E(id).onclick = () => move(dx, dy));

// 移动方向
function move(dx, dy) {
    if (S.running && S.dx === -dx && S.dy === -dy) return;
    [S.dx, S.dy] = [dx, dy];
    if (!S.running && (dx || dy)) startGame();
}

// 开始游戏
function startGame() {
    S.snake = [{ x: 10, y: 10 }];
    S.dx = 1; S.dy = 0;
    S.score = 0; S.speed = C.s;
    S.running = true;
    E('score').textContent = 0;
    E('startScreen').classList.add('hidden');
    E('gameOver').classList.add('hidden');
    genFood();
    clearInterval(S.loop);
    S.loop = setInterval(step, S.speed);
    draw();
}

// 游戏主循环
function step() {
    if (!S.running) return;
    const head = { x: S.snake[0].x + S.dx, y: S.snake[0].y + S.dy };

    // 碰撞检测
    if (head.x < 0 || head.x >= C.t || head.y < 0 || head.y >= C.t ||
        S.snake.some(s => s.x === head.x && s.y === head.y)) return gameOver();

    S.snake.unshift(head);

    // 吃食物
    if (head.x === S.food.x && head.y === S.food.y) {
        S.score += 10;
        E('score').textContent = S.score;
        genFood();
        if (S.score % 50 === 0 && S.speed > 80) {
            S.speed -= 5;
            clearInterval(S.loop);
            S.loop = setInterval(step, S.speed);
        }
    } else {
        S.snake.pop();
    }
    draw();
}

// 生成食物
function genFood() {
    let f;
    do {
        f = { x: Math.floor(Math.random() * C.t), y: Math.floor(Math.random() * C.t) };
    } while (S.snake.some(s => s.x === f.x && s.y === f.y));
    S.food = f;
}

// 绘制
function draw() {
    // 清屏
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 网格
    ctx.strokeStyle = '#16213e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= C.t; i++) {
        ctx.beginPath();
        ctx.moveTo(i * C.g, 0); ctx.lineTo(i * C.g, canvas.height);
        ctx.moveTo(0, i * C.g); ctx.lineTo(canvas.width, i * C.g);
        ctx.stroke();
    }

    // 蛇
    S.snake.forEach((s, i) => {
        ctx.fillStyle = i === 0 ? '#22c55e' : '#4ade80';
        ctx.fillRect(s.x * C.g + 2, s.y * C.g + 2, C.g - 4, C.g - 4);
    });

    // 食物
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(S.food.x * C.g + C.g / 2, S.food.y * C.g + C.g / 2, C.g / 2 - 2, 0, 6.28);
    ctx.fill();
}

// 游戏结束
function gameOver() {
    S.running = false;
    clearInterval(S.loop);
    if (S.score > S.high) {
        S.high = S.score;
        localStorage.setItem('snakeHighScore', S.high);
        E('highScore').textContent = S.high;
    }
    E('finalScore').textContent = S.score;
    E('gameOver').classList.remove('hidden');
}

draw();
