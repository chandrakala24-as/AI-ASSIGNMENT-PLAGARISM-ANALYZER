// Common Application Helpers & Auth Check

const API_BASE = "";

// Check if user is logged in
function checkAuth(allowedRoles = []) {
    const userStr = localStorage.getItem("currentUser");
    
    // If not logged in, redirect to login page
    if (!userStr) {
        if (!window.location.pathname.endsWith("index.html") && window.location.pathname !== "/") {
            window.location.href = "/static/index.html";
        }
        return null;
    }
    
    const user = JSON.parse(userStr);
    
    // Check if role is allowed
    if (allowedRoles.length > 0 && !allowedRoles.includes(user.role)) {
        // Unauthorized role, redirect to their proper dashboard
        redirectToDashboard(user.role);
        return null;
    }
    
    return user;
}

// Redirect user to their proper dashboard
function redirectToDashboard(role) {
    if (role === "admin") {
        window.location.href = "/static/admin.html";
    } else if (role === "teacher") {
        window.location.href = "/static/teacher.html";
    } else if (role === "student") {
        window.location.href = "/static/student.html";
    } else {
        window.location.href = "/static/index.html";
    }
}

// Log out user
function logout() {
    localStorage.removeItem("currentUser");
    window.location.href = "/static/index.html";
}

// Update sidebar profile card
function updateSidebarProfile(user) {
    const avatar = document.getElementById("user-avatar-initial");
    const nameEl = document.getElementById("user-profile-name");
    const roleEl = document.getElementById("user-profile-role");
    
    if (avatar && nameEl && roleEl && user) {
        avatar.textContent = user.full_name ? user.full_name.charAt(0).toUpperCase() : user.username.charAt(0).toUpperCase();
        nameEl.textContent = user.full_name || user.username;
        
        let roleText = user.role;
        if (user.role === "student" && user.section) {
            roleText += ` (${user.section})`;
        }
        roleEl.textContent = roleText;
    }
}

// Format date string
function formatDate(isoString) {
    if (!isoString) return "N/A";
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Format file size
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Helper to handle API requests with Form Data or JSON
async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            const errData = await response.json().catch(() => ({ detail: "Server error occurred" }));
            throw new Error(errData.detail || "Request failed");
        }
        return await response.json();
    } catch (error) {
        console.error(`API Request Error [${endpoint}]:`, error);
        throw error;
    }
}
