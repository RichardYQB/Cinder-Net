// 游戏配置
const config = {
    gridSize: 20,
    tileCount: 20,
    initialSpeed: 150
};

// 游戏状态
let gameState = {
    snake: [{ x: 10, y: 10 }],
    food: { x: 15, y: 15 },
    dx: 0,
    dy: 0,
    score: 0,
    highScore: localStorage.getItem('snakeHighScore') || 0,
    gameRunning: false,
    gameLoop: null,
    speed: config.initialSpeed
};

// DOM 元素
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('score');
const highScoreElement = document.getElementById('highScore');
const gameOverScreen = document.getElementById('gameOver');
const startScreen = document.getElementById('startScreen');
const finalScoreElement = document.getElementById('finalScore');
const restartBtn = document.getElementById('restartBtn');
const startBtn = document.getElementById('startBtn');

// 控制按钮
const upBtn = document.getElementById('upBtn');
const downBtn = document.getElementById('downBtn');
const leftBtn = document.getElementById('leftBtn');
const rightBtn = document.getElementById('rightBtn');

// 设置画布大小
canvas.width = config.tileCount * config.gridSize;
canvas.height = config.tileCount * config.gridSize;

// 初始化
function init() {
    highScoreElement.textContent = gameState.highScore;
    startScreen.classList.remove('hidden');
    gameOverScreen.classList.add('hidden');
    
    // 绑定事件
    startBtn.addEventListener('click', startGame);
    restartBtn.addEventListener('click', startGame);
    
    // 键盘控制
    document.addEventListener('keydown', handleKeyPress);
    
    // 触摸按钮控制
    upBtn.addEventListener('click', () => changeDirection(0, -1));
    downBtn.addEventListener('click', () => changeDirection(0, 1));
    leftBtn.addEventListener('click', () => changeDirection(-1, 0));
    rightBtn.addEventListener('click', () => changeDirection(1, 0));
    
    // 绘制初始画面
    draw();
}

// 开始游戏
function startGame() {
    // 重置游戏状态
    gameState.snake = [{ x: 10, y: 10 }];
    // 默认给一个初始方向，避免“开始了但蛇不动”看起来像没反应
    gameState.dx = 1;
    gameState.dy = 0;
    gameState.score = 0;
    gameState.speed = config.initialSpeed;
    gameState.gameRunning = true;
    
    // 生成新食物
    generateFood();
    
    // 更新UI
    scoreElement.textContent = gameState.score;
    startScreen.classList.add('hidden');
    gameOverScreen.classList.add('hidden');
    
    // 开始游戏循环
    if (gameState.gameLoop) {
        clearInterval(gameState.gameLoop);
    }
    gameState.gameLoop = setInterval(gameStep, gameState.speed);
    
    draw();
}

// 游戏主循环
function gameStep() {
    if (!gameState.gameRunning) return;
    
    // 移动蛇头
    const head = {
        x: gameState.snake[0].x + gameState.dx,
        y: gameState.snake[0].y + gameState.dy
    };
    
    // 检查墙壁碰撞
    if (head.x < 0 || head.x >= config.tileCount || 
        head.y < 0 || head.y >= config.tileCount) {
        gameOver();
        return;
    }
    
    // 检查自身碰撞
    if (gameState.snake.some(segment => segment.x === head.x && segment.y === head.y)) {
        gameOver();
        return;
    }
    
    gameState.snake.unshift(head);
    
    // 检查是否吃到食物
    if (head.x === gameState.food.x && head.y === gameState.food.y) {
        gameState.score += 10;
        scoreElement.textContent = gameState.score;
        generateFood();
        
        // 增加速度
        if (gameState.score % 50 === 0 && gameState.speed > 80) {
            gameState.speed -= 5;
            clearInterval(gameState.gameLoop);
            gameState.gameLoop = setInterval(gameStep, gameState.speed);
        }
    } else {
        gameState.snake.pop();
    }
    
    draw();
}

// 生成食物
function generateFood() {
    let newFood;
    do {
        newFood = {
            x: Math.floor(Math.random() * config.tileCount),
            y: Math.floor(Math.random() * config.tileCount)
        };
    } while (gameState.snake.some(segment => 
        segment.x === newFood.x && segment.y === newFood.y
    ));
    
    gameState.food = newFood;
}

// 改变方向
function changeDirection(dx, dy) {
    // 防止反向移动
    if (gameState.dx === -dx && gameState.dy === -dy) {
        return;
    }
    
    // 只有在游戏运行时才允许改变方向
    if (gameState.gameRunning || (gameState.dx === 0 && gameState.dy === 0)) {
        gameState.dx = dx;
        gameState.dy = dy;
        
        // 如果游戏还没开始，立即开始
        if (!gameState.gameRunning && (dx !== 0 || dy !== 0)) {
            startGame();
        }
    }
}

// 处理键盘输入
function handleKeyPress(e) {
    const key = e.key.toLowerCase();
    
    switch(key) {
        case 'arrowup':
        case 'w':
            e.preventDefault();
            changeDirection(0, -1);
            break;
        case 'arrowdown':
        case 's':
            e.preventDefault();
            changeDirection(0, 1);
            break;
        case 'arrowleft':
        case 'a':
            e.preventDefault();
            changeDirection(-1, 0);
            break;
        case 'arrowright':
        case 'd':
            e.preventDefault();
            changeDirection(1, 0);
            break;
    }
}

// 绘制游戏
function draw() {
    // 清空画布
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制网格线（可选）
    ctx.strokeStyle = '#16213e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= config.tileCount; i++) {
        ctx.beginPath();
        ctx.moveTo(i * config.gridSize, 0);
        ctx.lineTo(i * config.gridSize, canvas.height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * config.gridSize);
        ctx.lineTo(canvas.width, i * config.gridSize);
        ctx.stroke();
    }
    
    // 绘制蛇
    ctx.fillStyle = '#4ade80';
    gameState.snake.forEach((segment, index) => {
        if (index === 0) {
            // 蛇头用不同颜色
            ctx.fillStyle = '#22c55e';
        } else {
            ctx.fillStyle = '#4ade80';
        }
        
        ctx.fillRect(
            segment.x * config.gridSize + 2,
            segment.y * config.gridSize + 2,
            config.gridSize - 4,
            config.gridSize - 4
        );
    });
    
    // 绘制食物
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(
        gameState.food.x * config.gridSize + config.gridSize / 2,
        gameState.food.y * config.gridSize + config.gridSize / 2,
        config.gridSize / 2 - 2,
        0,
        2 * Math.PI
    );
    ctx.fill();
}

// 游戏结束
function gameOver() {
    gameState.gameRunning = false;
    clearInterval(gameState.gameLoop);
    
    // 更新最高分
    if (gameState.score > gameState.highScore) {
        gameState.highScore = gameState.score;
        localStorage.setItem('snakeHighScore', gameState.highScore);
        highScoreElement.textContent = gameState.highScore;
    }
    
    finalScoreElement.textContent = gameState.score;
    gameOverScreen.classList.remove('hidden');
}

// 初始化游戏
init();
