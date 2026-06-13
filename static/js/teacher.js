// Teacher Dashboard JavaScript Logic

let teacherUser = null;
let activeSection = "CY2A";
let activeAssignments = [];
let selectedAssignmentId = null;
let activeSubmissions = [];
let selectedSubmission = null;
let teacherPollInterval = null;

// Charts instances
let chartGrades = null;
let chartRisk = null;
let chartCorrelation = null;

document.addEventListener("DOMContentLoaded", () => {
    teacherUser = checkAuth(["teacher"]);
    if (teacherUser) {
        updateSidebarProfile(teacherUser);
        
        // Setup default date for due date field to tomorrow
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        document.getElementById("new-asg-due").value = tomorrow.toISOString().split('T')[0];
        
        loadTeacherAssignments();
    }
});

// Switch sidebar tabs
function switchTeacherTab(tabName) {
    const subTab = document.getElementById("tab-section-submissions");
    const analyticsTab = document.getElementById("tab-section-analytics");
    
    // Toggle active link CSS
    const links = document.querySelectorAll(".sidebar-link");
    links.forEach(l => l.classList.remove("active"));
    
    if (tabName === "submissions") {
        subTab.style.display = "block";
        analyticsTab.style.display = "none";
        event.currentTarget.classList.add("active");
        loadTeacherAssignments();
    } else if (tabName === "analytics") {
        subTab.style.display = "none";
        analyticsTab.style.display = "block";
        event.currentTarget.classList.add("active");
        if (teacherPollInterval) {
            clearInterval(teacherPollInterval);
            teacherPollInterval = null;
        }
        loadAnalytics();
    }
}

// Section pills filters
function filterBySection(section) {
    activeSection = section;
    
    const pills = document.querySelectorAll(".section-pill");
    pills.forEach(p => p.classList.remove("active"));
    
    // Highlight selected pill
    event.currentTarget.classList.add("active");
    
    // Update assignment create form section default
    document.getElementById("new-asg-section").value = section;
    
    // Reload assignments & submissions for this section
    renderAssignmentsDropdown();
}

// Fetch all assignments created by teacher
async function loadTeacherAssignments() {
    try {
        activeAssignments = await apiRequest(`/api/teacher/assignments?teacher_id=${teacherUser.user_id}`);
        renderAssignmentsDropdown();
    } catch (err) {
        console.error("Failed to load assignments:", err);
    }
}

// Render dropdown filtered by active section
function renderAssignmentsDropdown() {
    const select = document.getElementById("select-teacher-assignment");
    select.innerHTML = '<option value="" disabled selected>-- Select Assignment Prompt --</option>';
    
    const sectionAssignments = activeAssignments.filter(a => a.class_section === activeSection);
    
    sectionAssignments.forEach(asg => {
        const opt = document.createElement("option");
        opt.value = asg.id;
        opt.textContent = `${asg.title} (Due: ${asg.due_date ? new Date(asg.due_date).toLocaleDateString() : 'N/A'})`;
        select.appendChild(opt);
    });
    
    // If the previously selected assignment is still in this section, keep it
    if (selectedAssignmentId && sectionAssignments.some(a => a.id === selectedAssignmentId)) {
        select.value = selectedAssignmentId;
        loadSubmissionsList();
    } else {
        selectedAssignmentId = null;
        document.getElementById("btn-batch-scan").setAttribute("disabled", "true");
        document.getElementById("teacher-submissions-table-body").innerHTML = 
            `<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">Select an assignment from the dropdown to view submissions.</td></tr>`;
        if (teacherPollInterval) {
            clearInterval(teacherPollInterval);
            teacherPollInterval = null;
        }
    }
}

function handleAssignmentSelect() {
    selectedAssignmentId = parseInt(document.getElementById("select-teacher-assignment").value);
    if (selectedAssignmentId) {
        document.getElementById("btn-batch-scan").removeAttribute("disabled");
        loadSubmissionsList();
        
        if (teacherPollInterval) clearInterval(teacherPollInterval);
        teacherPollInterval = setInterval(loadSubmissionsList, 3000);
    } else {
        if (teacherPollInterval) {
            clearInterval(teacherPollInterval);
            teacherPollInterval = null;
        }
    }
}

// Fetch student submissions for selected assignment
async function loadSubmissionsList() {
    if (!selectedAssignmentId) return;
    
    try {
        activeSubmissions = await apiRequest(`/api/teacher/submissions?assignment_id=${selectedAssignmentId}`);
        const tableBody = document.getElementById("teacher-submissions-table-body");
        tableBody.innerHTML = "";
        
        if (activeSubmissions.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">No student reports submitted for this assignment yet.</td></tr>`;
            return;
        }
        
        activeSubmissions.forEach(sub => {
            let plagText = '<span style="color: var(--text-muted); font-size: 0.8rem;">Not Scanned</span>';
            if (sub.overall_plagiarism_pct !== null) {
                const pct = sub.overall_plagiarism_pct;
                let badgeClass = "badge-green";
                if (pct >= 50) badgeClass = "badge-red";
                else if (pct >= 20) badgeClass = "badge-yellow";
                
                plagText = `<span class="badge ${badgeClass}" style="cursor:pointer;" onclick="openGraderModal('${sub.submission_id}')">${pct}% Plag</span>`;
            }
            
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td style="font-weight: 600; color: var(--text-primary);">${sub.student_name}</td>
                <td style="font-family: monospace; font-size: 0.8rem; max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${sub.file_name}</td>
                <td>${formatDate(sub.submitted_at)}</td>
                <td>${plagText}</td>
                <td style="font-weight: 700;">
                    ${sub.marks !== null ? `<span style="color: var(--accent-green);">${sub.marks} / 100</span>` : '<span style="color: var(--text-muted); font-size: 0.85rem;">Ungraded</span>'}
                </td>
                <td style="text-align: right; display: flex; justify-content: flex-end; gap: 8px;">
                    <button class="btn btn-secondary btn-icon" onclick="runPlagiarismScan('${sub.submission_id}', '${sub.student_name}')" title="Scan Assignment">
                        <i class="fa-solid fa-wand-magic-sparkles" style="color: var(--accent-cyan);"></i>
                    </button>
                    <button class="btn btn-primary" style="padding: 6px 12px; font-size: 0.8rem;" onclick="openGraderModal('${sub.submission_id}')" ${sub.overall_plagiarism_pct === null ? 'disabled' : ''}>
                        <i class="fa-solid fa-pen-nib"></i> Grade
                    </button>
                </td>
            `;
            tableBody.appendChild(tr);
        });
    } catch (err) {
        console.error("Failed to load submissions:", err);
    }
}

// Create New Assignment Prompt
async function handleCreateAssignment(event) {
    event.preventDefault();
    
    const title = document.getElementById("new-asg-title").value.trim();
    const desc = document.getElementById("new-asg-desc").value.trim();
    const section = document.getElementById("new-asg-section").value;
    const dueDate = document.getElementById("new-asg-due").value;
    const alertBox = document.getElementById("create-asg-alert");
    
    alertBox.style.display = "none";
    
    try {
        const formData = new FormData();
        formData.append("title", title);
        formData.append("description", desc);
        formData.append("class_section", section);
        formData.append("due_date", dueDate);
        formData.append("teacher_id", teacherUser.user_id);
        
        const response = await fetch("/api/teacher/assignments", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to create assignment");
        }
        
        alertBox.className = "auth-alert auth-alert-success";
        alertBox.textContent = `Assignment "${title}" published successfully for section ${section}!`;
        alertBox.style.display = "block";
        
        // Reset form fields
        document.getElementById("new-asg-title").value = "";
        document.getElementById("new-asg-desc").value = "";
        
        // Reload assignments list
        await loadTeacherAssignments();
        
        setTimeout(() => {
            alertBox.style.display = "none";
        }, 3000);
        
    } catch (err) {
        alertBox.className = "auth-alert auth-alert-error";
        alertBox.textContent = err.message;
        alertBox.style.display = "block";
    }
}

// Immersive scanning animation helper
function showScanningModal(title) {
    document.getElementById("scanning-title").textContent = title;
    document.getElementById("scanning-progress-bar").style.width = "0%";
    const logBox = document.getElementById("scanning-log");
    logBox.innerHTML = "";
    
    document.getElementById("scanning-modal").classList.add("active");
}

function addLogLine(text, delay = 0) {
    return new Promise(resolve => {
        setTimeout(() => {
            const logBox = document.getElementById("scanning-log");
            const div = document.createElement("div");
            div.className = "scan-log-line";
            div.innerHTML = `<span style="color: var(--accent-cyan); font-weight: bold;">&gt;</span> ${text}`;
            logBox.appendChild(div);
            logBox.scrollTop = logBox.scrollHeight;
            resolve();
        }, delay);
    });
}

function updateScanProgressBar(percent) {
    document.getElementById("scanning-progress-bar").style.width = `${percent}%`;
}

function closeScanningModal() {
    document.getElementById("scanning-modal").classList.remove("active");
}

// Run single Plagiarism Scan
async function runPlagiarismScan(submissionId, studentName) {
    showScanningModal(`Scanning Assignment: ${studentName}`);
    
    // Trigger the backend API call immediately in background
    const scanPromise = fetch(`/api/teacher/scan/${submissionId}`, { method: "POST" })
        .then(res => {
            if (!res.ok) throw new Error("Plagiarism engine encounterd an error.");
            return res.json();
        });
        
    // Animate logging sequentially
    try {
        await addLogLine("[SYS] Initializing local scanning protocols...", 100);
        updateScanProgressBar(15);
        
        await addLogLine("[SYS] Loading file characters text block...", 300);
        updateScanProgressBar(28);
        
        await addLogLine("[AI] Constructing Character 5-Grams & Word 3-Grams...", 400);
        updateScanProgressBar(40);
        
        await addLogLine("[AI] Fitting TF-IDF vectors & Cosine similarity models...", 400);
        updateScanProgressBar(55);
        
        await addLogLine("[AI] Querying BERT semantic encoder embeddings...", 500);
        updateScanProgressBar(70);
        
        await addLogLine("[AI] Fingerprinting document winnowing hash maps...", 450);
        updateScanProgressBar(82);
        
        await addLogLine("[SYS] Crawling simulated internet wiki index databases...", 400);
        updateScanProgressBar(92);
        
        // Wait for backend to finish if it hasn't already
        const scanResult = await scanPromise;
        
        await addLogLine(`[SYS] Scan complete. Plagiarism Index calculated: ${scanResult.results.overall_plagiarism}%!`, 300);
        updateScanProgressBar(100);
        
        setTimeout(async () => {
            closeScanningModal();
            // Reload submission rows
            await loadSubmissionsList();
            // Open the grading interface automatically!
            openGraderModal(submissionId);
        }, 800);
        
    } catch (err) {
        await addLogLine(`[ERROR] Scanning process aborted: ${err.message}`, 100);
        updateScanProgressBar(100);
        
        setTimeout(() => {
            closeScanningModal();
            alert("Error scanning submission: " + err.message);
        }, 1500);
    }
}

// Run batch section scan
async function triggerBatchScan() {
    if (!selectedAssignmentId) return;
    if (activeSubmissions.length === 0) {
        alert("No submissions available to scan.");
        return;
    }
    
    showScanningModal(`Batch Scanning Section: ${activeSection}`);
    
    try {
        await addLogLine(`[SYS] Initializing batch scan for ${activeSubmissions.length} assignments...`, 150);
        
        for (let i = 0; i < activeSubmissions.length; i++) {
            const sub = activeSubmissions[i];
            const progressPct = Math.round(((i) / activeSubmissions.length) * 100);
            updateScanProgressBar(progressPct);
            
            await addLogLine(`Scanning student submission [${i + 1}/${activeSubmissions.length}]: ${sub.student_name}...`, 200);
            
            // Execute scan API synchronously for stability in batch
            const response = await fetch(`/api/teacher/scan/${sub.submission_id}`, { method: "POST" });
            if (!response.ok) {
                await addLogLine(`[WARN] Failed to scan submission for ${sub.student_name}.`, 100);
            } else {
                const data = await response.json();
                await addLogLine(`Scanned ${sub.student_name} successfully. Overal similarity: ${data.results.overall_plagiarism}%.`, 100);
            }
        }
        
        updateScanProgressBar(100);
        await addLogLine("[SYS] Batch scan process finalized. Database indexes updated.", 200);
        
        setTimeout(async () => {
            closeScanningModal();
            await loadSubmissionsList();
        }, 1200);
        
    } catch (err) {
        closeScanningModal();
        alert("Batch scanning failed: " + err.message);
    }
}

// Grader Modal Controls
function openGraderModal(submissionId) {
    selectedSubmission = activeSubmissions.find(s => s.submission_id === submissionId);
    if (!selectedSubmission || !selectedSubmission.detailed_report) return;
    
    const rep = selectedSubmission.detailed_report;
    const text = selectedSubmission.extracted_text || "";
    
    // Set title and circular scores
    document.getElementById("grader-modal-title").textContent = `Plagiarism Grader - Student: ${selectedSubmission.student_name}`;
    
    document.getElementById("grader-val-overall").textContent = `${Math.round(selectedSubmission.overall_plagiarism_pct)}%`;
    document.getElementById("grader-progress-overall").style.background = `conic-gradient(var(--accent-red) ${selectedSubmission.overall_plagiarism_pct * 3.6}deg, rgba(255, 255, 255, 0.05) 0deg)`;
    
    document.getElementById("grader-val-peer").textContent = `${Math.round(selectedSubmission.peer_similarity_pct)}%`;
    document.getElementById("grader-progress-peer").style.background = `conic-gradient(var(--accent-violet) ${selectedSubmission.peer_similarity_pct * 3.6}deg, rgba(255, 255, 255, 0.05) 0deg)`;
    
    document.getElementById("grader-val-internet").textContent = `${Math.round(selectedSubmission.internet_similarity_pct)}%`;
    document.getElementById("grader-progress-internet").style.background = `conic-gradient(var(--accent-cyan) ${selectedSubmission.internet_similarity_pct * 3.6}deg, rgba(255, 255, 255, 0.05) 0deg)`;
    
    // Render Highlighted Text
    const docViewer = document.getElementById("grader-doc-viewer");
    docViewer.innerHTML = generateHighlightedText(text, rep.peer_matches, rep.internet_matches);
    
    // Render Flagged Sources List
    const sourcesList = document.getElementById("grader-sources-list");
    sourcesList.innerHTML = "";
    
    let hasMatches = false;
    
    // Peer matches UI list
    if (rep.peer_matches && rep.peer_matches.length > 0) {
        hasMatches = true;
        rep.peer_matches.forEach(m => {
            const div = document.createElement("div");
            div.style.cssText = "padding: 8px; border-left: 3px solid var(--accent-violet); background: rgba(138, 43, 226, 0.05); border-radius: 0 6px 6px 0; font-size: 0.8rem;";
            div.innerHTML = `
                <div style="display:flex; justify-content:space-between; font-weight:600; color:var(--text-primary);">
                    <span>Student: ${m.student_name} (${m.student_section})</span>
                    <span style="color:var(--accent-violet);">${m.similarity_pct}% Copy</span>
                </div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:2px;">
                    Algorithm breakdown: TF-IDF:${m.scores.tfidf}% | Winnowing:${m.scores.winnowing}% | BERT:${m.scores.bert}%
                </div>
            `;
            sourcesList.appendChild(div);
        });
    }
    
    // Internet matches UI list
    if (rep.internet_matches && rep.internet_matches.length > 0) {
        hasMatches = true;
        rep.internet_matches.forEach(m => {
            const div = document.createElement("div");
            div.style.cssText = "padding: 8px; border-left: 3px solid var(--accent-cyan); background: rgba(0, 242, 254, 0.05); border-radius: 0 6px 6px 0; font-size: 0.8rem;";
            div.innerHTML = `
                <div style="display:flex; justify-content:space-between; font-weight:600; color:var(--text-primary);">
                    <span>Web: <a href="${m.url}" target="_blank" style="color:var(--accent-cyan);">${m.title}</a></span>
                    <span style="color:var(--accent-cyan);">${m.similarity_pct}% Copy</span>
                </div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:2px; word-break: break-all;">
                    URL: ${m.url}
                </div>
            `;
            sourcesList.appendChild(div);
        });
    }
    
    if (!hasMatches) {
        sourcesList.innerHTML = `<div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 10px;">No plagiarism sources flagged.</div>`;
    }
    
    // Set grading input values
    document.getElementById("grader-marks").value = selectedSubmission.marks !== null ? selectedSubmission.marks : "";
    document.getElementById("grader-feedback").value = selectedSubmission.feedback !== null ? selectedSubmission.feedback : "";
    document.getElementById("grader-alert").style.display = "none";
    
    // Open Modal
    document.getElementById("grader-modal").classList.add("active");
}

function closeGraderModal() {
    document.getElementById("grader-modal").classList.remove("active");
}

// Handle Grades Submission
async function handleGradeSubmit(event) {
    event.preventDefault();
    if (!selectedSubmission) return;
    
    const marks = document.getElementById("grader-marks").value;
    const feedback = document.getElementById("grader-feedback").value.trim();
    const alertBox = document.getElementById("grader-alert");
    
    alertBox.style.display = "none";
    
    try {
        const formData = new FormData();
        formData.append("marks", marks);
        formData.append("feedback", feedback);
        
        const response = await fetch(`/api/teacher/grade/${selectedSubmission.submission_id}`, {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to submit marks");
        }
        
        alertBox.className = "auth-alert auth-alert-success";
        alertBox.textContent = "Marks and feedback recorded successfully!";
        alertBox.style.display = "block";
        
        // Reload submissions list
        await loadSubmissionsList();
        
        // Close modal after delay
        setTimeout(() => {
            closeGraderModal();
        }, 1200);
        
    } catch (err) {
        alertBox.className = "auth-alert auth-alert-error";
        alertBox.textContent = err.message;
        alertBox.style.display = "block";
    }
}

// Render Plagiarism Highlights (helper copy from student.js)
function generateHighlightedText(text, peerMatches, internetMatches) {
    if (!text) return '<span style="color: var(--text-muted); font-style: italic;">[No content extracted from document]</span>';
    
    const highlightType = new Array(text.length).fill(0);
    const highlightSource = new Array(text.length).fill("");
    
    if (internetMatches) {
        internetMatches.forEach(src => {
            if (src.matched_spans) {
                src.matched_spans.forEach(span => {
                    const start = span.start1;
                    const end = span.end1;
                    for (let i = start; i < end && i < highlightType.length; i++) {
                        highlightType[i] = 2;
                        highlightSource[i] = `${src.title} (${src.url})`;
                    }
                });
            }
        });
    }
    
    if (peerMatches) {
        peerMatches.forEach(peer => {
            if (peer.matched_spans) {
                peer.matched_spans.forEach(span => {
                    const start = span.start1;
                    const end = span.end1;
                    for (let i = start; i < end && i < highlightType.length; i++) {
                        highlightType[i] = 1;
                        highlightSource[i] = `${peer.student_name} (${peer.student_section})`;
                    }
                });
            }
        });
    }
    
    let html = [];
    let i = 0;
    while (i < text.length) {
        const type = highlightType[i];
        const src = highlightSource[i];
        
        if (type === 0) {
            let start = i;
            while (i < text.length && highlightType[i] === 0) i++;
            html.push(escapeHtml(text.substring(start, i)));
        } else {
            let start = i;
            while (i < text.length && highlightType[i] === type && highlightSource[i] === src) i++;
            const className = type === 1 ? "highlight-peer" : "highlight-internet";
            html.push(`<span class="${className}" data-source="${escapeHtml(src)}">${escapeHtml(text.substring(start, i))}</span>`);
        }
    }
    return html.join("");
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

// Fetch and render analytics charts using Chart.js
async function loadAnalytics() {
    try {
        const data = await apiRequest(`/api/teacher/analytics?teacher_id=${teacherUser.user_id}`);
        
        const sections = data.section_data.map(d => d.section);
        const avgMarks = data.section_data.map(d => d.avg_marks);
        const avgPlag = data.section_data.map(d => d.avg_plagiarism);
        
        // 1. GRADE AVERAGES BAR CHART
        if (chartGrades) chartGrades.destroy();
        chartGrades = new Chart(document.getElementById("chart-grades"), {
            type: 'bar',
            data: {
                labels: sections,
                datasets: [
                    {
                        label: 'Average Score',
                        data: avgMarks,
                        backgroundColor: 'rgba(0, 242, 254, 0.4)',
                        borderColor: 'var(--accent-cyan)',
                        borderWidth: 2,
                        borderRadius: 6
                    },
                    {
                        label: 'Average Plagiarism %',
                        data: avgPlag,
                        backgroundColor: 'rgba(138, 43, 226, 0.4)',
                        borderColor: 'var(--accent-violet)',
                        borderWidth: 2,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#8e94a8', font: { family: 'Inter' } } }
                },
                scales: {
                    x: { ticks: { color: '#8e94a8' }, grid: { color: 'rgba(255,255,255,0.03)' } },
                    y: { min: 0, max: 100, ticks: { color: '#8e94a8' }, grid: { color: 'rgba(255,255,255,0.03)' } }
                }
            }
        });
        
        // 2. RISK LEVELS DOUGHNUT CHART
        const risk = data.risk_distribution;
        if (chartRisk) chartRisk.destroy();
        chartRisk = new Chart(document.getElementById("chart-risk"), {
            type: 'doughnut',
            data: {
                labels: ['Low Risk (<20%)', 'Medium Risk (20-50%)', 'High Risk (>50%)'],
                datasets: [{
                    data: [risk.low, risk.medium, risk.high],
                    backgroundColor: [
                        'rgba(0, 230, 118, 0.4)',  // Green
                        'rgba(255, 235, 59, 0.4)',  // Yellow
                        'rgba(255, 23, 68, 0.4)'    // Red
                    ],
                    borderColor: [
                        'var(--accent-green)',
                        'var(--accent-yellow)',
                        'var(--accent-red)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { color: '#8e94a8', font: { family: 'Inter' } } }
                }
            }
        });
        
        // 3. SCATTER CORRELATION CHART
        const correlationData = data.correlations.map(c => ({ x: c.plagiarism, y: c.marks, label: c.student_name }));
        if (chartCorrelation) chartCorrelation.destroy();
        chartCorrelation = new Chart(document.getElementById("chart-correlation"), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Students',
                    data: correlationData,
                    backgroundColor: 'rgba(138, 43, 226, 0.6)',
                    borderColor: 'var(--accent-violet)',
                    borderWidth: 1,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const pt = context.raw;
                                return `${pt.label}: Plag: ${pt.x}%, Mark: ${pt.y}/100`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Plagiarism Rate (%)', color: '#8e94a8' },
                        ticks: { color: '#8e94a8' },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        min: 0,
                        max: 100
                    },
                    y: {
                        title: { display: true, text: 'Marks Allotted', color: '#8e94a8' },
                        ticks: { color: '#8e94a8' },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
        
    } catch (err) {
        console.error("Failed to render charts:", err);
    }
}
