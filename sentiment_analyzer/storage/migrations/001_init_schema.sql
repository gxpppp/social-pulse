-- 数据库初始化脚本
-- 版本: 001
-- 描述: 创建核心表结构、索引和触发器
-- 数据库: SQLite

-- 启用外键约束
PRAGMA foreign_keys = ON;

-- ============================================
-- 用户表 (users)
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(64) NOT NULL UNIQUE,
    platform VARCHAR(32) NOT NULL,
    username VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    bio TEXT,
    avatar_url VARCHAR(512),
    avatar_hash VARCHAR(64),
    registered_at DATETIME,
    followers_count INTEGER DEFAULT 0,
    friends_count INTEGER DEFAULT 0,
    posts_count INTEGER DEFAULT 0,
    verified BOOLEAN DEFAULT 0,
    first_seen DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_updated DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    is_suspicious BOOLEAN DEFAULT 0,
    suspicious_score REAL,
    extra_data TEXT,
    created_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 用户表索引
CREATE INDEX IF NOT EXISTS idx_users_platform_username ON users(platform, username);
CREATE INDEX IF NOT EXISTS idx_users_platform ON users(platform);
CREATE INDEX IF NOT EXISTS idx_users_suspicious ON users(is_suspicious);
CREATE INDEX IF NOT EXISTS idx_users_first_seen ON users(first_seen);
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);

-- 用户表触发器: 自动更新 last_updated
CREATE TRIGGER IF NOT EXISTS trg_users_updated_at
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    UPDATE users SET last_updated = datetime('now', 'localtime') WHERE id = OLD.id;
END;

-- ============================================
-- 帖子表 (posts)
-- ============================================
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id VARCHAR(64) NOT NULL UNIQUE,
    user_id VARCHAR(64) NOT NULL,
    platform VARCHAR(32) NOT NULL,
    content TEXT,
    language VARCHAR(10),
    posted_at DATETIME NOT NULL,
    likes_count INTEGER DEFAULT 0,
    shares_count INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    views_count INTEGER DEFAULT 0,
    hashtags TEXT,
    mentions TEXT,
    urls TEXT,
    media TEXT,
    parent_post_id VARCHAR(64),
    is_retweet BOOLEAN DEFAULT 0,
    is_reply BOOLEAN DEFAULT 0,
    collected_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    content_hash VARCHAR(64),
    sentiment_score REAL,
    extra_data TEXT,
    created_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 帖子表索引
CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id);
CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);
CREATE INDEX IF NOT EXISTS idx_posts_posted_at ON posts(posted_at);
CREATE INDEX IF NOT EXISTS idx_posts_collected_at ON posts(collected_at);
CREATE INDEX IF NOT EXISTS idx_posts_parent ON posts(parent_post_id);
CREATE INDEX IF NOT EXISTS idx_posts_content_hash ON posts(content_hash);
CREATE INDEX IF NOT EXISTS idx_posts_post_id ON posts(post_id);

-- 帖子表触发器: 自动更新 updated_at
CREATE TRIGGER IF NOT EXISTS trg_posts_updated_at
AFTER UPDATE ON posts
FOR EACH ROW
BEGIN
    UPDATE posts SET updated_at = datetime('now', 'localtime') WHERE id = OLD.id;
END;

-- ============================================
-- 互动表 (interactions)
-- ============================================
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id VARCHAR(128) NOT NULL UNIQUE,
    user_id VARCHAR(64) NOT NULL,
    post_id VARCHAR(64) NOT NULL,
    platform VARCHAR(32) NOT NULL,
    interaction_type VARCHAR(32) NOT NULL,
    content TEXT,
    interacted_at DATETIME NOT NULL,
    collected_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    extra_data TEXT,
    created_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE CASCADE
);

-- 互动表索引
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_post_id ON interactions(post_id);
CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_interactions_platform ON interactions(platform);
CREATE INDEX IF NOT EXISTS idx_interactions_interacted_at ON interactions(interacted_at);
CREATE INDEX IF NOT EXISTS idx_interactions_user_post_type ON interactions(user_id, post_id, interaction_type);
CREATE INDEX IF NOT EXISTS idx_interactions_interaction_id ON interactions(interaction_id);

-- 互动表触发器: 自动更新 updated_at
CREATE TRIGGER IF NOT EXISTS trg_interactions_updated_at
AFTER UPDATE ON interactions
FOR EACH ROW
BEGIN
    UPDATE interactions SET updated_at = datetime('now', 'localtime') WHERE id = OLD.id;
END;

-- ============================================
-- 采集任务表 (crawl_tasks)
-- ============================================
CREATE TABLE IF NOT EXISTS crawl_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id VARCHAR(64) NOT NULL UNIQUE,
    platform VARCHAR(32) NOT NULL,
    task_type VARCHAR(32) NOT NULL,
    target VARCHAR(512) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    config TEXT,
    result_count INTEGER DEFAULT 0,
    error_message TEXT,
    started_at DATETIME,
    completed_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);

-- 采集任务表索引
CREATE INDEX IF NOT EXISTS idx_crawl_tasks_platform ON crawl_tasks(platform);
CREATE INDEX IF NOT EXISTS idx_crawl_tasks_status ON crawl_tasks(status);
CREATE INDEX IF NOT EXISTS idx_crawl_tasks_task_type ON crawl_tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_crawl_tasks_priority ON crawl_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_crawl_tasks_created_at ON crawl_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_crawl_tasks_task_id ON crawl_tasks(task_id);

-- 采集任务表触发器: 自动更新 updated_at
CREATE TRIGGER IF NOT EXISTS trg_crawl_tasks_updated_at
AFTER UPDATE ON crawl_tasks
FOR EACH ROW
BEGIN
    UPDATE crawl_tasks SET updated_at = datetime('now', 'localtime') WHERE id = OLD.id;
END;

-- ============================================
-- 用户特征表 (user_features)
-- ============================================
CREATE TABLE IF NOT EXISTS user_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(64) NOT NULL UNIQUE,
    platform VARCHAR(32) NOT NULL,
    
    -- 时序行为特征
    daily_post_avg REAL,
    daily_post_std REAL,
    hour_entropy REAL,
    night_activity_ratio REAL,
    weekend_activity_ratio REAL,
    
    -- 内容生成特征
    content_similarity_avg REAL,
    topic_entropy REAL,
    sentiment_variance REAL,
    avg_text_length REAL,
    url_ratio REAL,
    mention_ratio REAL,
    
    -- 账号元数据特征
    follower_ratio REAL,
    account_age_days INTEGER,
    profile_completeness REAL,
    
    -- 社交网络特征
    degree_centrality REAL,
    betweenness_centrality REAL,
    clustering_coefficient REAL,
    
    -- 预测结果
    anomaly_score REAL,
    predicted_label VARCHAR(32),
    confidence_score REAL,
    
    computed_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    feature_version VARCHAR(32) DEFAULT '1.0',
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 用户特征表索引
CREATE INDEX IF NOT EXISTS idx_user_features_user_id ON user_features(user_id);
CREATE INDEX IF NOT EXISTS idx_user_features_platform ON user_features(platform);
CREATE INDEX IF NOT EXISTS idx_user_features_anomaly_score ON user_features(anomaly_score);
CREATE INDEX IF NOT EXISTS idx_user_features_predicted_label ON user_features(predicted_label);

-- ============================================
-- 系统日志表 (system_logs)
-- ============================================
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_level VARCHAR(16) NOT NULL,
    module VARCHAR(64) NOT NULL,
    message TEXT NOT NULL,
    details TEXT,
    created_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 系统日志表索引
CREATE INDEX IF NOT EXISTS idx_system_logs_log_level ON system_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_module ON system_logs(module);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);

-- ============================================
-- 迁移版本记录表
-- ============================================
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(32) PRIMARY KEY,
    applied_at DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    description TEXT
);

-- 记录当前版本
INSERT OR IGNORE INTO schema_migrations (version, description)
VALUES ('001', '初始化数据库表结构');

-- ============================================
-- 视图定义
-- ============================================

-- 用户统计视图
CREATE VIEW IF NOT EXISTS v_user_stats AS
SELECT 
    u.user_id,
    u.platform,
    u.username,
    u.followers_count,
    u.friends_count,
    u.posts_count,
    u.is_suspicious,
    u.suspicious_score,
    COUNT(DISTINCT p.id) as actual_posts_count,
    COUNT(DISTINCT i.id) as interaction_count,
    MAX(p.posted_at) as last_post_at
FROM users u
LEFT JOIN posts p ON u.user_id = p.user_id
LEFT JOIN interactions i ON u.user_id = i.user_id
GROUP BY u.user_id;

-- 帖子统计视图
CREATE VIEW IF NOT EXISTS v_post_stats AS
SELECT 
    p.post_id,
    p.platform,
    p.user_id,
    u.username,
    p.posted_at,
    p.likes_count,
    p.shares_count,
    p.comments_count,
    p.views_count,
    (p.likes_count + p.shares_count * 2 + p.comments_count * 3) as engagement_score,
    COUNT(DISTINCT i.id) as interaction_count
FROM posts p
LEFT JOIN users u ON p.user_id = u.user_id
LEFT JOIN interactions i ON p.post_id = i.post_id
GROUP BY p.post_id;

-- 采集任务统计视图
CREATE VIEW IF NOT EXISTS v_task_stats AS
SELECT 
    platform,
    task_type,
    status,
    COUNT(*) as task_count,
    SUM(result_count) as total_results,
    AVG(result_count) as avg_results,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count
FROM crawl_tasks
GROUP BY platform, task_type, status;

-- ============================================
-- 初始数据
-- ============================================

-- 插入默认配置（如果需要）
-- INSERT INTO system_config (key, value, description) VALUES ...

-- 完成提示
SELECT 'Database schema initialized successfully!' as message;
