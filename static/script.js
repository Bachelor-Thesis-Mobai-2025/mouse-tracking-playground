let trajectoryData = null;
let currentQuestion = null;
let questionStartTime = null;
let selectedAnswer = null;
let microPauses = 0;
let totalPauseTime = 0;
let lastMovementTime = null;
let buttonOrder = Math.random() < 0.5; // Randomize initial button order
let firstMovementTime = null;
let firstDecisionMade = false;
let firstDecisionIndex = -1;
let answerChanges = 0;
let buttonHoverData = {
    yes: { enterTime: null, totalTime: 0, enterCount: 0 },
    no: { enterTime: null, totalTime: 0, enterCount: 0 }
};
let directionChanges = 0;
let lastDirection = null;
let lastVelocity = null;
let velocityBuffer = [];
const ACCEL_WINDOW_SIZE = 5; // Use 5 samples for smoother acceleration

// Grab references
const yesBtn = document.getElementById('yes-btn');
const noBtn = document.getElementById('no-btn');
const nextBtn = document.getElementById('next-btn');
const questionElement = document.getElementById('question');
const btnGroup = document.querySelector('.btn-group');

// Fixed-rate sampling at 100Hz (10ms)
const samplingInterval = 1000 / 100; // 10ms for 100Hz

// Current mouse position
let currentMouseX = 0;
let currentMouseY = 0;
let lastRecordedX = 0;
let lastRecordedY = 0;
let lastDx = 0;
let lastDy = 0;

// Load location embeds
const locationEmbeds = {
    // Norway locations
    "Norway": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d7115129.1005240865!2d7.202227919814455!3d64.19044446471571!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x461268458f4de5bf%3A0xa1b03b9db864d02b!2sNorge!5e0!3m2!1sno!2sno!4v1742225138899!5m2!1sno!2sno",
    "NTNU": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d1947.0953831666425!2d10.679612413300815!3d60.78973439317518!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x4641da14f48bc6d1%3A0x15d7b34504988672!2zTlROVSBww6UgR2rDuHZpaw!5e0!3m2!1sno!2sno!4v1742225884390!5m2!1sno!2sno",
    "Gj\u00F8vik": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d31680.92327955288!2d10.676129899999999!3d60.798508!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x4641da7f7d25d825%3A0xc834e9351bd371f1!2sGj%C3%B8vik!5e0!3m2!1sno!2sno!4v1711064304382!5m2!1sno!2sno",
    "Innlandet": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d1999520.267789213!2d10.49408859677242!3d61.26669287075171!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x4613202593a930b7%3A0x4613202593a931b5!2sInnlandet!5e0!3m2!1sno!2sno!4v1711064362211!5m2!1sno!2sno",

    // United States locations
    "USA": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d24314527.50780894!2d-102.58805748934844!3d40.13885294325144!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x54eab584e432360b%3A0x1c3bb99243deb742!2sUSA!5e0!3m2!1sno!2sno!4v1742225936787!5m2!1sno!2sno",
    "Berkeley": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d50660.14164260444!2d-122.30440033250237!3d37.87223135536998!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x80857c3254417ca3%3A0x1f0ce75c7cfefe47!2sBerkeley%2C%20CA%2C%20USA!5e0!3m2!1sno!2sno!4v1711064428032!5m2!1sno!2sno",
    "California": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d12443468.11099838!2d-122.97043661879774!3d36.778261015535724!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x80859a6d00690021%3A0x4a501367f076adff!2sCalifornien%2C%20USA!5e0!3m2!1sno!2sno!4v1711064461141!5m2!1sno!2sno",
    "University of California, Berkeley": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3149.532814939858!2d-122.25804328751286!3d37.87121830648002!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x808f7718c522d7c1%3A0xda8034ea3b6b3289!2sUniversity%20of%20California!5e0!3m2!1sno!2sno!4v1742226174344!5m2!1sno!2sno",

    // Australian locations
    "Australia": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d34368450.22012237!2d134.48828173214755!3d-25.27433291133309!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x2b2bfd076787c5df%3A0x538267a1955b1352!2sAustralia!5e0!3m2!1sno!2sno!4v1711064593220!5m2!1sno!2sno",
    "Royal Melbourne Institute of Technology": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3152.2194357913054!2d144.96135831278156!3d-37.808328833612855!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad642cb0a2ff0fb%3A0xed6e6acedcefb31c!2sRMIT%20University!5e0!3m2!1sno!2sno!4v1742224523627!5m2!1sno!2sno",
    "Melbourne": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d402590.52635753667!2d144.72282398041585!3d-37.971563335846504!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad646b5d2ba4df7%3A0x4045675218ccd90!2sMelbourne%20Victoria%2C%20Australia!5e0!3m2!1sno!2sno!4v1742226652455!5m2!1sno!2sno",
    "Victoria": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3285885.6074694144!2d140.45994787418059!3d-36.46063347963703!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad4314b7e18954f%3A0x5a4efce2be829534!2sVictoria%2C%20Australia!5e0!3m2!1sno!2sno!4v1742226621015!5m2!1sno!2sno",
    };

// Initialize trajectory data_new structure
function initTrajectoryData() {
    const currentTime = Date.now();
    trajectoryData = {
        sequence_id: currentTime,
        label: "unknown", // Will be set by the server
        trajectory_metrics: {
            decision_path_efficiency: 0,    // Efficiency up to first decision
            final_decision_path_efficiency: 0, // Efficiency up to final decision
            total_time: 0,
            hesitation_time: 0,
            time_to_first_movement: 0,
            hesitation_count: 0,
            direction_changes: 0,
            hover_time: 0,
            hover_count: 0,
            total_pause_time: 0,
            pause_count: 0,
            answer_changes: 0    // Number of times answer changed
        },
        trajectory: []
    };

    // Reset metrics
    microPauses = 0;
    totalPauseTime = 0;
    lastMovementTime = null;
    firstMovementTime = null;
    firstDecisionMade = false;
    firstDecisionIndex = -1;
    answerChanges = 0;
    buttonHoverData = {
        yes: { enterTime: null, totalTime: 0, enterCount: 0 },
        no: { enterTime: null, totalTime: 0, enterCount: 0 }
    };
    directionChanges = 0;
    lastDirection = null;
    lastVelocity = null;
    velocityBuffer = [];
}

// Calculate smoothed acceleration from velocity buffer
function calculateAcceleration(velocity) {
    // Update velocity buffer
    velocityBuffer.push(velocity);
    if (velocityBuffer.length > ACCEL_WINDOW_SIZE) {
        velocityBuffer.shift();
    }

    // Calculate acceleration
    let acceleration = 0;

    if (velocityBuffer.length >= 3) {
        // Use linear regression to find the slope of velocity over time
        const x = Array.from({length: velocityBuffer.length}, (_, i) => i);
        const y = velocityBuffer;

        // Calculate means
        const xMean = x.reduce((a, b) => a + b, 0) / x.length;
        const yMean = y.reduce((a, b) => a + b, 0) / y.length;

        // Calculate slope (acceleration)
        const numerator = x.reduce((acc, xi, i) => acc + (xi - xMean) * (y[i] - yMean), 0);
        const denominator = x.reduce((acc, xi) => acc + Math.pow(xi - xMean, 2), 0);

        // Convert to acceleration in pixels/sec²
        if (denominator !== 0) {
            acceleration = numerator / denominator / (samplingInterval / 1000);
        }
    } else if (lastVelocity !== null) {
        // Fallback to simple calculation when not enough samples
        acceleration = (velocity - lastVelocity) / (samplingInterval / 1000);
    }

    // Update lastVelocity for next calculation
    lastVelocity = velocity;

    return acceleration;
}

// Calculate curvature for a point
function calculateCurvature(p1, p2, p3) {
    if (!p1 || !p2 || !p3) return 0;

    // Convert to vectors
    const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };

    // Calculate the cross product magnitude
    const crossProduct = v1.x * v2.y - v1.y * v2.x;

    // Calculate magnitudes
    const v1Mag = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const v2Mag = Math.sqrt(v2.x * v2.x + v2.y * v2.y);

    // Avoid division by zero
    if (v1Mag * v2Mag === 0) return 0;

    // Calculate curvature (K = |v1 × v2| / (|v1| * |v2|))
    return Math.abs(crossProduct) / (v1Mag * v2Mag);
}

// Calculate jerk
function calculateJerk(prev2, prev1, current) {
    if (!prev2 || !prev1 || !current) return 0;

    // Calculate acceleration for previous time step
    const dt = samplingInterval / 1000; // Time step in seconds
    const a1x = (prev1.dx - prev2.dx) / dt;
    const a1y = (prev1.dy - prev2.dy) / dt;

    // Calculate acceleration for current time step
    const a2x = (current.dx - prev1.dx) / dt;
    const a2y = (current.dy - prev1.dy) / dt;

    // Calculate jerk (rate of change of acceleration)
    const jerkX = (a2x - a1x) / dt;
    const jerkY = (a2y - a1y) / dt;

    // Magnitude of jerk
    return Math.sqrt(jerkX * jerkX + jerkY * jerkY);
}

function calculateCurvatureJerk(dx, dy) {
    const trajectory = trajectoryData.trajectory;

    // Calculate curvature
    let curvature = 0;
    if (trajectory.length >= 2) {
        const p1 = trajectory.length >= 3 ? trajectory[trajectory.length - 3] : null;
        const p2 = trajectory[trajectory.length - 2];

        if (p1) {
            curvature = calculateCurvature(
                {x: p1.x, y: p1.y},
                {x: p2.x, y: p2.y},
                {x: currentMouseX, y: currentMouseY}
            );
        }
    }

    // Calculate jerk
    let jerk = 0;
    if (trajectory.length >= 2) {
        const prev2 = trajectory.length >= 3 ? trajectory[trajectory.length - 3] : null;
        const prev1 = trajectory[trajectory.length - 2];
        const current = {
            dx: dx,
            dy: dy
        };

        if (prev2) {
            jerk = calculateJerk(prev2, prev1, current);
        }
    }
    return {curvature, jerk};
}

// Calculate path efficiency
function calculatePathEfficiency(trajectory) {
    if (trajectory.length < 2) return 1.0;

    const start = trajectory[0];
    const end = trajectory[trajectory.length - 1];

    // Direct distance (straight line)
    const directDistance = Math.sqrt(
        (end.x - start.x) ** 2 + (end.y - start.y) ** 2
    );

    // Actual path length
    let pathLength = 0;
    for (let i = 1; i < trajectory.length; i++) {
        const segmentLength = Math.sqrt(
            (trajectory[i].x - trajectory[i-1].x) ** 2 +
            (trajectory[i].y - trajectory[i-1].y) ** 2
        );
        pathLength += segmentLength;
    }

    // Avoid division by zero
    if (pathLength === 0) return 1.0;

    return directDistance / pathLength;
}

// Button hover handlers
function onYesEnter() {
    if (buttonHoverData.yes.enterTime === null) {
        buttonHoverData.yes.enterTime = Date.now();
        buttonHoverData.yes.enterCount++;
    }
}

function onYesLeave() {
    if (buttonHoverData.yes.enterTime !== null) {
        buttonHoverData.yes.totalTime += Date.now() - buttonHoverData.yes.enterTime;
        buttonHoverData.yes.enterTime = null;
    }
}

function onNoEnter() {
    if (buttonHoverData.no.enterTime === null) {
        buttonHoverData.no.enterTime = Date.now();
        buttonHoverData.no.enterCount++;
    }
}

function onNoLeave() {
    if (buttonHoverData.no.enterTime !== null) {
        buttonHoverData.no.totalTime += Date.now() - buttonHoverData.no.enterTime;
        buttonHoverData.no.enterTime = null;
    }
}

// Setup button hover tracking
function setupButtonHoverTracking() {
    // Remove previous event listeners if any
    yesBtn.removeEventListener('mouseenter', onYesEnter);
    yesBtn.removeEventListener('mouseleave', onYesLeave);
    noBtn.removeEventListener('mouseenter', onNoEnter);
    noBtn.removeEventListener('mouseleave', onNoLeave);

    // Re-add event listeners
    yesBtn.addEventListener('mouseenter', onYesEnter);
    yesBtn.addEventListener('mouseleave', onYesLeave);
    noBtn.addEventListener('mouseenter', onNoEnter);
    noBtn.addEventListener('mouseleave', onNoLeave);
}

// Human clicks handlers
function handleYesClick() {
    if (!yesBtn.classList.contains('selected')) {
        selectedAnswer = 1; // Yes = 1
        logClick();
        yesBtn.classList.add('selected');
        noBtn.classList.remove('selected');
    }
}

function handleNoClick() {
    if (!noBtn.classList.contains('selected')) {
        selectedAnswer = 0; // No = 0
        logClick();
        noBtn.classList.add('selected');
        yesBtn.classList.remove('selected');
    }
}

// Log a click
function logClick() {
    const currentTime = Date.now();

    // Calculate dx and dy from last recorded position
    const dx = currentMouseX - lastRecordedX;
    const dy = currentMouseY - lastRecordedY;

    // Displacement (pixels)
    const displacement = Math.sqrt(dx**2 + dy**2);

    // Time elapsed (seconds)
    const timeElapsed = samplingInterval / 1000;

    // Velocity (pixels per second)
    const velocity = displacement / timeElapsed;

    // Calculate acceleration using our smoothing function
    const acceleration = calculateAcceleration(velocity);

    // Get trajectory for curvature and jerk calculations
    let {curvature, jerk} = calculateCurvatureJerk(dx, dy);

    const entry = {
        timestamp: currentTime,
        x: currentMouseX,
        y: currentMouseY,
        dx: dx,
        dy: dy,
        velocity: velocity,
        acceleration: acceleration,
        curvature: curvature, // Now calculated instead of 0
        jerk: jerk,           // Now calculated instead of 0
        click: 1
    };

    // Add entry to trajectory
    trajectoryData.trajectory.push(entry);

    // If this is the first decision, calculate and store decision path efficiency
    if (!firstDecisionMade) {
        firstDecisionMade = true;
        firstDecisionIndex = trajectoryData.trajectory.length - 1;

        // Calculate path efficiency up to this point
        const pathToFirstDecision = trajectoryData.trajectory.slice(0, firstDecisionIndex + 1);
        trajectoryData.trajectory_metrics.decision_path_efficiency =
            calculatePathEfficiency(pathToFirstDecision);
    } else {
        // If not the first decision, increment answer changes counter
        answerChanges++;
    }
}

// Calculate final trajectory metrics
function calculateTrajectoryMetrics() {
    if (!trajectoryData) return;

    const trajectory = trajectoryData.trajectory;

    // Find the last click event in the trajectory
    let lastClickIndex = -1;
    for (let i = trajectory.length - 1; i >= 0; i--) {
        if (trajectory[i].click === 1) {
            lastClickIndex = i;
            break;
        }
    }

    // If no click was found, use the entire trajectory
    // Otherwise, only use the trajectory up to the last click
    const pathForEfficiency = lastClickIndex >= 0 ?
        trajectory.slice(0, lastClickIndex + 1) :
        trajectory;

    // Calculate path efficiency up to the last click
    trajectoryData.trajectory_metrics.final_decision_path_efficiency =
        calculatePathEfficiency(pathForEfficiency);

    // If no first decision was recorded, but we have a final decision,
    // copy the final path efficiency to the initial one
    if (!firstDecisionMade && pathForEfficiency.length > 0) {
        trajectoryData.trajectory_metrics.decision_path_efficiency =
            trajectoryData.trajectory_metrics.final_decision_path_efficiency;
    }

    // Total time
    trajectoryData.trajectory_metrics.total_time =
        (trajectory[trajectory.length - 1].timestamp - trajectory[0].timestamp) / 1000;

    // Time to first movement
    if (firstMovementTime !== null) {
        trajectoryData.trajectory_metrics.time_to_first_movement =
            (firstMovementTime - questionStartTime) / 1000;
    }

    // Combined button hover data_new
    trajectoryData.trajectory_metrics.hover_time =
        (buttonHoverData.yes.totalTime + buttonHoverData.no.totalTime) / 1000;
    trajectoryData.trajectory_metrics.hover_count =
        buttonHoverData.yes.enterCount + buttonHoverData.no.enterCount;

    // Total hesitation time (same as button hover time)
    trajectoryData.trajectory_metrics.hesitation_time = trajectoryData.trajectory_metrics.hover_time;
    trajectoryData.trajectory_metrics.hesitation_count = trajectoryData.trajectory_metrics.hover_count;

    // Pauses
    trajectoryData.trajectory_metrics.total_pause_time = totalPauseTime / 1000;
    trajectoryData.trajectory_metrics.pause_count = microPauses;

    // Direction changes
    trajectoryData.trajectory_metrics.direction_changes = directionChanges;

    // Answer changes
    trajectoryData.trajectory_metrics.answer_changes = answerChanges;
}

// Save data_new
function saveData() {
    // Don't save if no answer selected
    if (selectedAnswer === null) {
        console.warn("No answer selected, not saving data_new");
        return;
    }

    // Calculate final metrics
    calculateTrajectoryMetrics();

    // Add answer to data_new
    trajectoryData.answer = selectedAnswer; // 1 for yes, 0 for no

    fetch('/log_data', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(trajectoryData)
    })
        .then(response => response.json())
        .then(data => {
            if (data.status !== "success") {
                console.error("Failed to save data_new:", data.message);
            }
        })
        .catch(error => {
            console.error("Error saving data_new:", error);
        });

    // Reset selected answer
    selectedAnswer = null;
}

// Swap button positions
function swapButtonPositions() {
    buttonOrder = !buttonOrder;

    if (buttonOrder) {
        btnGroup.innerHTML = '';
        btnGroup.appendChild(yesBtn);
        btnGroup.appendChild(noBtn);
    } else {
        btnGroup.innerHTML = '';
        btnGroup.appendChild(noBtn);
        btnGroup.appendChild(yesBtn);
    }

    // Re-add event listeners to buttons
    yesBtn.addEventListener('click', handleYesClick);
    noBtn.addEventListener('click', handleNoClick);

    // Re-add hover tracking
    setupButtonHoverTracking();
}

// Shake animation for buttons
function shakeElement(element) {
    element.classList.add('shake-animation');
    setTimeout(() => {
        element.classList.remove('shake-animation');
    }, 500);
}

// Fetch question from server
function fetchQuestion() {
    const url = `/get_question`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.complete) {
                // Experiment is complete
                questionElement.innerText = data.instruction;
                // Hide the answer buttons and next button
                yesBtn.style.display = 'none';
                noBtn.style.display = 'none';
                nextBtn.style.display = 'none';
                // Hide map container
                document.getElementById('map-container').style.display = 'none';
                return;
            }

            // Check if this is an instruction screen
            if (data.isInstruction) {
                // Display instruction
                questionElement.innerText = data.instruction;
                // Hide answer buttons for instructions
                yesBtn.style.display = 'none';
                noBtn.style.display = 'none';
                // Show next button to continue
                nextBtn.style.display = 'block';
                // Hide map container
                document.getElementById('map-container').style.display = 'none';
                // Reset timer
                questionStartTime = Date.now();
                return;
            }

            // Normal question display
            currentQuestion = data.question;
            questionElement.innerText = data.question;

            // Initialize trajectory data_new structure
            initTrajectoryData();

            // Swap button positions randomly
            swapButtonPositions();

            // Check if we should show a map for this question
            const mapContainer = document.getElementById('map-container');
            const mapIframe = document.getElementById('location-map');
            let mapFound = false;

            // Check each location against the question
            for (const [location, embedUrl] of Object.entries(locationEmbeds)) {
                if (data.question.includes(location)) {
                    mapIframe.src = embedUrl;
                    mapContainer.style.display = 'block';
                    mapFound = true;
                    break;
                }
            }

            // Hide map if no matching location
            if (!mapFound) {
                mapContainer.style.display = 'none';
            }

            // Reset button states
            yesBtn.classList.remove('selected');
            noBtn.classList.remove('selected');

            // Reset timer
            questionStartTime = Date.now();

            // Make sure buttons are visible
            yesBtn.style.display = 'block';
            noBtn.style.display = 'block';
            nextBtn.style.display = 'block';

            // Reset selected answer
            selectedAnswer = null;
        })
        .catch(error => {
            console.error("Error fetching question:", error);
        });
}

// Update timer display
function updateTimer() {
    if (questionStartTime) {
        const elapsed = Math.floor((Date.now() - questionStartTime) / 1000);
        document.getElementById('timer').textContent = `Time on question: ${elapsed}s`;
    }
}

// CSS for shake animation
const shakeStyles = `
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.shake-animation {
    animation: shake 0.5s ease-in-out;
}
`;

// Main event listeners
document.addEventListener('mousemove', function(event) {
    const now = Date.now();

    // Record first movement time
    if (currentMouseX !== event.clientX || currentMouseY !== event.clientY) {
        if (firstMovementTime === null && questionStartTime !== null) {
            firstMovementTime = now;
        }

        // Reset pause detection
        if (lastMovementTime !== null) {
            const pauseDuration = now - lastMovementTime;
            if (pauseDuration >= 10) { // Pause threshold (10ms)
                microPauses++;
                totalPauseTime += pauseDuration;
            }
        }

        lastMovementTime = now;
    }

    // Just update the current position
    currentMouseX = event.clientX;
    currentMouseY = event.clientY;
});

// Sample mouse movement at fixed rate
setInterval(function() {
    if (!trajectoryData) return;

    const currentTime = Date.now();

    // Calculate dx and dy from last recorded position
    const dx = currentMouseX - lastRecordedX;
    const dy = currentMouseY - lastRecordedY;

    // Displacement (pixels)
    const displacement = Math.sqrt(dx**2 + dy**2);

    // Time elapsed (seconds)
    const timeElapsed = samplingInterval / 1000;

    // Velocity (pixels per second)
    const velocity = displacement / timeElapsed;

    // Calculate smoothed acceleration
    const acceleration = calculateAcceleration(velocity);

    // Detect direction changes
    if (dx !== 0 || dy !== 0) {
        // Calculate movement direction (in 8 directions)
        const angle = Math.atan2(dy, dx) * 180 / Math.PI;
        const direction = Math.round(angle / 45) * 45;

        // Check if direction changed significantly
        if (lastDirection !== null && direction !== lastDirection) {
            directionChanges++;
        }
        lastDirection = direction;
    }

    // Get trajectory length for curvature and jerk calculations
    let {curvature, jerk} = calculateCurvatureJerk(dx, dy);

    const entry = {
        timestamp: currentTime,
        x: currentMouseX,
        y: currentMouseY,
        dx: dx,
        dy: dy,
        velocity: velocity,
        acceleration: acceleration,
        curvature: curvature,
        jerk: jerk,
        click: 0
    };

    // Only update the last recorded position AFTER recording the entry
    lastRecordedX = currentMouseX;
    lastRecordedY = currentMouseY;
    lastDx = dx;
    lastDy = dy;

    // Add to trajectory
    trajectoryData.trajectory.push(entry);
}, samplingInterval);

// Update timer display at regular intervals
setInterval(updateTimer, 1000);

// Add click handlers
yesBtn.addEventListener('click', handleYesClick);
noBtn.addEventListener('click', handleNoClick);

// Add styles to the document if not already present
const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = shakeStyles;
document.head.appendChild(styleSheet);

// Next button click handler
nextBtn.addEventListener('click', () => {
    if (yesBtn.style.display !== 'none' && noBtn.style.display !== 'none') {
        saveData();
    }
    if (yesBtn.classList.contains('selected')
        || noBtn.classList.contains('selected')
        || yesBtn.style.display === 'none'
        && noBtn.style.display === 'none') {
        fetchQuestion();
    } else {
        // Shake both buttons to indicate selection is needed
        shakeElement(yesBtn);
        shakeElement(noBtn);
    }
});

// Initialize button hover tracking
setupButtonHoverTracking();

// Initial load
fetchQuestion();