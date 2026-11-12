// pm2 process for the WhatsMeal camera worker on the VPS. Its own name; it listens on no port.
module.exports = {
  apps: [
    {
      name: 'whatsmeal-vision',
      cwd: __dirname,
      script: './run.sh',
      interpreter: 'bash',
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      autorestart: true,
      max_restarts: 50,
      min_uptime: '30s',
      restart_delay: 10000,
      // The models take about 1.3 GB. Above this something is wrong: start again rather than starve the other apps.
      max_memory_restart: '2500M',
      kill_timeout: 10000,
      out_file: './logs/pm2-out.log',
      error_file: './logs/pm2-error.log',
      merge_logs: true,
      time: true,
    },
  ],
};
