/* AICOS control-room session helper. Tokens are intentionally kept in
 * sessionStorage (tab lifetime) rather than localStorage or URL query strings.
 */
(function (global) {
  'use strict';
  var KEY = 'aicosSessionToken';
  var USER = 'aicosSessionUser';
  function token() { try { return sessionStorage.getItem(KEY) || ''; } catch (e) { return ''; } }
  function user() { try { return JSON.parse(sessionStorage.getItem(USER) || 'null'); } catch (e) { return null; } }
  function setSession(t, u) {
    try { sessionStorage.setItem(KEY, t || ''); sessionStorage.setItem(USER, JSON.stringify(u || null)); } catch (e) {}
  }
  function clear() { try { sessionStorage.removeItem(KEY); sessionStorage.removeItem(USER); } catch (e) {} }
  function baseUrl(v) { return String(v || window.location.origin).replace(/\/+$/, ''); }
  function headers(extra) {
    var h = Object.assign({}, extra || {}), t = token();
    if (t) h.Authorization = 'Bearer ' + t;
    return h;
  }
  async function login(base, username, password) {
    var r = await fetch(baseUrl(base) + '/api/auth/login', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({username: username, password: password})
    });
    if (!r.ok) throw new Error('Login failed (HTTP ' + r.status + ')');
    var d = await r.json(); setSession(d.token, d.user); return d.user;
  }
  async function logout(base) {
    var t = token();
    if (t) { try { await fetch(baseUrl(base) + '/api/auth/logout', {method:'POST', headers:headers()}); } catch(e) {} }
    clear();
  }
  global.AicosAuth = {token:token, user:user, setSession:setSession, clear:clear, headers:headers, login:login, logout:logout, baseUrl:baseUrl};
})(window);
