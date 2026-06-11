/* ============================================================
   AEGIS-OS — backend endpoint configuration
   ------------------------------------------------------------
   The frontend (this static shell) talks to the FastAPI backend
   over a WebSocket (/aegis/stream) + REST. When the backend
   serves this page itself (local dev, or the Render URL) leave
   AEGIS_BACKEND empty → same origin.

   For a SPLIT deploy (Vercel frontend + Render backend) set this
   to your Render service URL, then redeploy Vercel:

       window.AEGIS_BACKEND = "https://sepsis-backend.onrender.com";

   You can also override live without redeploying:
       localStorage.setItem('aegis_backend', 'https://…onrender.com')
   ============================================================ */
window.AEGIS_BACKEND = "";   // <-- set to your Render URL for the Vercel deploy

window.AEGIS = (function () {
  function base() {
    try { var o = localStorage.getItem("aegis_backend"); if (o) return o.replace(/\/+$/, ""); } catch (e) {}
    var b = (window.AEGIS_BACKEND || "").replace(/\/+$/, "");
    return b || location.origin;
  }
  return {
    base: base,
    http: function (path) { return base() + (path || ""); },
    ws: function () { return base().replace(/^http/, "ws") + "/aegis/stream"; },
  };
})();
