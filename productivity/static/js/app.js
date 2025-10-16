// Productivity Tracker Dashboard JavaScript

class ProductivityDashboard {
    constructor() {
        this.socket = io();
        this.isTracking = false;
        this.startTime = null;
        this.liveChart = null;
        this.summaryChart = null;
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        
        this.initializeCharts();
        this.setupEventListeners();
        this.setupSocketListeners();
        this.startDataRefresh();
    }

    initializeCharts() {
        // Live data chart
        const liveCtx = document.getElementById('liveChart').getContext('2d');
        this.liveChart = new Chart(liveCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'People Inside',
                        data: [],
                        borderColor: '#17a2b8',
                        backgroundColor: 'rgba(23, 162, 184, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Active People',
                        data: [],
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Phone Usage',
                        data: [],
                        borderColor: '#6c757d',
                        backgroundColor: 'rgba(108, 117, 125, 0.1)',
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        beginAtZero: true,
                        max: 20
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });

        // Summary chart
        const summaryCtx = document.getElementById('summaryChart').getContext('2d');
        this.summaryChart = new Chart(summaryCtx, {
            type: 'bar',
            data: {
                labels: ['People Inside', 'Active', 'Loitering', 'Phone Usage'],
                datasets: [{
                    label: 'Average Count',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(23, 162, 184, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(108, 117, 125, 0.8)'
                    ],
                    borderColor: [
                        '#17a2b8',
                        '#ffc107',
                        '#dc3545',
                        '#6c757d'
                    ],
                    borderWidth: 2,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    setupEventListeners() {
        // Start/Stop buttons
        document.getElementById('startBtn').addEventListener('click', () => {
            this.startTracking();
        });

        document.getElementById('stopBtn').addEventListener('click', () => {
            this.stopTracking();
        });

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Page is hidden, reduce update frequency
                this.pauseUpdates();
            } else {
                // Page is visible, resume normal updates
                this.resumeUpdates();
            }
        });
    }

    setupSocketListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus('Connected', 'success');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus('Disconnected', 'danger');
        });

        this.socket.on('live_data', (data) => {
            this.updateLiveData(data);
        });

        this.socket.on('status', (data) => {
            console.log('Status update:', data.message);
        });
    }

    async startTracking() {
        try {
            const response = await fetch('/start_tracking', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isTracking = true;
                this.startTime = new Date();
                this.frameCount = 0;
                
                // Update UI
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('systemStatus').textContent = 'Running';
                document.getElementById('systemStatus').className = 'badge bg-success';
                
                // Start video stream
                this.startVideoStream();
                
                // Update connection status
                this.updateConnectionStatus('Connected', 'success');
                
                console.log('Tracking started successfully');
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error starting tracking:', error);
            alert('Failed to start tracking: ' + error.message);
        }
    }

    async stopTracking() {
        try {
            const response = await fetch('/stop_tracking', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isTracking = false;
                this.startTime = null;
                
                // Update UI
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('systemStatus').textContent = 'Stopped';
                document.getElementById('systemStatus').className = 'badge bg-secondary';
                
                // Stop video stream
                this.stopVideoStream();
                
                // Update connection status
                this.updateConnectionStatus('Disconnected', 'secondary');
                
                console.log('Tracking stopped successfully');
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error stopping tracking:', error);
            alert('Failed to stop tracking: ' + error.message);
        }
    }

    startVideoStream() {
        const videoElement = document.getElementById('videoStream');
        const placeholder = document.getElementById('videoPlaceholder');
        
        videoElement.src = '/video_feed?' + new Date().getTime();
        videoElement.style.display = 'block';
        placeholder.style.display = 'none';
        
        // Handle video load events
        videoElement.onload = () => {
            this.frameCount++;
            this.updateFPS();
        };
        
        videoElement.onerror = () => {
            console.error('Video stream error');
            this.updateConnectionStatus('Stream Error', 'danger');
        };
    }

    stopVideoStream() {
        const videoElement = document.getElementById('videoStream');
        const placeholder = document.getElementById('videoPlaceholder');
        
        videoElement.src = '';
        videoElement.style.display = 'none';
        placeholder.style.display = 'block';
        
        // Reset counters
        document.getElementById('fpsCounter').textContent = '0';
        document.getElementById('resolution').textContent = 'N/A';
    }

    updateConnectionStatus(status, type) {
        const statusElement = document.getElementById('connectionStatus');
        statusElement.textContent = status;
        statusElement.className = `badge bg-${type}`;
    }

    updateFPS() {
        const now = Date.now();
        const timeDiff = (now - this.lastFrameTime) / 1000;
        const fps = timeDiff > 0 ? (1 / timeDiff).toFixed(1) : 0;
        
        document.getElementById('fpsCounter').textContent = fps;
        this.lastFrameTime = now;
    }

    updateLiveData(data) {
        // Update metric cards with animation
        this.updateMetricCard('peopleInside', data.inside_count || 0);
        this.updateMetricCard('activePeople', data.active_count || 0);
        this.updateMetricCard('loiteringPeople', data.loiter_count || 0);
        this.updateMetricCard('phoneUsage', data.phone_in_use || 0);
        
        // Update last update time
        const now = new Date();
        document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
        
        // Update processing FPS if available
        if (data.processing_fps) {
            document.getElementById('processingRate').textContent = data.processing_fps + ' fps';
        }
        
        // Update live chart
        this.updateLiveChart(data);
        
        // Update uptime
        this.updateUptime();
    }

    updateMetricCard(elementId, value) {
        const element = document.getElementById(elementId);
        const currentValue = parseInt(element.textContent) || 0;
        
        if (currentValue !== value) {
            element.textContent = value;
            element.classList.add('updated');
            setTimeout(() => {
                element.classList.remove('updated');
            }, 300);
        }
    }

    updateLiveChart(data) {
        const chart = this.liveChart;
        const now = new Date().toLocaleTimeString();
        
        // Add new data point
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(data.inside_count || 0);
        chart.data.datasets[1].data.push(data.active_count || 0);
        chart.data.datasets[2].data.push(data.phone_in_use || 0);
        
        // Keep only last 15 data points for better performance
        if (chart.data.labels.length > 15) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }
        
        // Update chart with no animation for better performance
        chart.update('none');
    }

    updateUptime() {
        if (this.startTime) {
            const now = new Date();
            const diff = now - this.startTime;
            const hours = Math.floor(diff / (1000 * 60 * 60));
            const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((diff % (1000 * 60)) / 1000);
            
            const uptime = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            document.getElementById('systemUptime').textContent = uptime;
        }
    }

    async refreshSummaryData() {
        try {
            const response = await fetch('/api/summary');
            const data = await response.json();
            
            if (data.current_session) {
                // Update summary statistics
                document.getElementById('avgPeopleInside').textContent = data.current_session.avg_people_inside || 0;
                document.getElementById('avgActivePeople').textContent = data.current_session.avg_active_people || 0;
                document.getElementById('avgLoitering').textContent = data.current_session.avg_loitering || 0;
                document.getElementById('avgPhoneUsage').textContent = data.current_session.avg_phone_usage || 0;
                document.getElementById('totalDataPoints').textContent = data.current_session.total_data_points || 0;
                
                // Update summary chart
                this.summaryChart.data.datasets[0].data = [
                    data.current_session.avg_people_inside || 0,
                    data.current_session.avg_active_people || 0,
                    data.current_session.avg_loitering || 0,
                    data.current_session.avg_phone_usage || 0
                ];
                this.summaryChart.update();
            }
            
            // Update system info
            document.getElementById('totalFrames').textContent = this.frameCount;
            const processingRate = this.isTracking ? document.getElementById('fpsCounter').textContent + ' fps' : '0 fps';
            document.getElementById('processingRate').textContent = processingRate;
            
        } catch (error) {
            console.error('Error refreshing summary data:', error);
        }
    }

    startDataRefresh() {
        // Refresh summary data every 10 seconds
        setInterval(() => {
            this.refreshSummaryData();
        }, 10000);
        
        // Update uptime every second
        setInterval(() => {
            this.updateUptime();
        }, 1000);
        
        // Initial load
        this.refreshSummaryData();
    }

    pauseUpdates() {
        // Reduce update frequency when page is not visible
        console.log('Pausing updates - page not visible');
    }

    resumeUpdates() {
        // Resume normal update frequency when page becomes visible
        console.log('Resuming updates - page visible');
        this.refreshSummaryData();
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ProductivityDashboard();
});
