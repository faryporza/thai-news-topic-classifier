import { useState, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:5000'

function App() {
  const [headline, setHeadline] = useState('')
  const [body, setBody] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')

  // ตรวจสอบสถานะ API เมื่อโหลดหน้า
  useEffect(() => {
    checkHealth()
    fetchModelInfo()
  }, [])

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`)
      const data = await res.json()
      setApiStatus(data.status === 'healthy' ? 'online' : 'offline')
    } catch {
      setApiStatus('offline')
    }
  }

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_URL}/model/info`)
      const data = await res.json()
      setModelInfo(data)
    } catch {
      console.log('Could not fetch model info')
    }
  }

  const handlePredict = async () => {
    if (!headline.trim() && !body.trim()) {
      setError('กรุณากรอก Headline หรือ Body อย่างน้อย 1 อย่าง')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ headline, body })
      })

      const data = await res.json()

      if (res.ok) {
        setResult(data)
      } else {
        setError(data.message || data.error || 'เกิดข้อผิดพลาด')
      }
    } catch {
      setError('ไม่สามารถเชื่อมต่อกับ API ได้ กรุณาตรวจสอบว่า Backend กำลังทำงาน')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setHeadline('')
    setBody('')
    setResult(null)
    setError(null)
  }

  const getTopicIcon = (topic) => {
    switch (topic) {
      case 'Business': return '💼'
      case 'SciTech': return '🔬'
      case 'World': return '🌍'
      default: return '📰'
    }
  }

  const getTopicColor = (topic) => {
    switch (topic) {
      case 'Business': return '#22c55e'
      case 'SciTech': return '#3b82f6'
      case 'World': return '#f59e0b'
      default: return '#6b7280'
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>🇹🇭 Thai News Topic Classifier</h1>
          <p>ระบบจำแนกหมวดหมู่ข่าวภาษาไทย</p>
          <div className={`status-badge ${apiStatus}`}>
            <span className="status-dot"></span>
            API: {apiStatus === 'online' ? 'ออนไลน์' : apiStatus === 'offline' ? 'ออฟไลน์' : 'กำลังตรวจสอบ...'}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main">
        <div className="container">
          {/* Input Section */}
          <section className="card input-section">
            <h2>📝 กรอกข้อมูลข่าว</h2>

            <div className="form-group">
              <label htmlFor="headline">Headline (พาดหัวข่าว)</label>
              <input
                id="headline"
                type="text"
                placeholder="เช่น บริษัทเทคโนโลยีเปิดตัวผลิตภัณฑ์ใหม่..."
                value={headline}
                onChange={(e) => setHeadline(e.target.value)}
                disabled={loading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="body">Body (เนื้อหาข่าว)</label>
              <textarea
                id="body"
                rows="6"
                placeholder="เช่น รายงานระบุว่าเทคโนโลยีดังกล่าวอาจช่วยเพิ่มประสิทธิภาพการใช้งาน..."
                value={body}
                onChange={(e) => setBody(e.target.value)}
                disabled={loading}
              />
            </div>

            <div className="button-group">
              <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={loading || apiStatus !== 'online'}
              >
                {loading ? '⏳ กำลังทำนาย...' : '🔮 Predict'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={handleClear}
                disabled={loading}
              >
                🗑️ ล้างข้อมูล
              </button>
            </div>

            {error && (
              <div className="error-message">
                ❌ {error}
              </div>
            )}
          </section>

          {/* Result Section */}
          {result && (
            <section className="card result-section">
              <h2>📊 ผลการทำนาย</h2>

              <div className="result-main" style={{ borderColor: getTopicColor(result.label) }}>
                <div className="result-icon">{getTopicIcon(result.label)}</div>
                <div className="result-label" style={{ color: getTopicColor(result.label) }}>
                  {result.label}
                </div>
                <div className="result-confidence">
                  Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                </div>
              </div>

              <div className="probabilities">
                <h3>📈 ความน่าจะเป็นแต่ละหมวด</h3>
                {Object.entries(result.probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([topic, prob]) => (
                    <div key={topic} className="probability-bar">
                      <div className="probability-label">
                        {getTopicIcon(topic)} {topic}
                      </div>
                      <div className="probability-track">
                        <div
                          className="probability-fill"
                          style={{
                            width: `${prob * 100}%`,
                            backgroundColor: getTopicColor(topic)
                          }}
                        />
                      </div>
                      <div className="probability-value">{(prob * 100).toFixed(1)}%</div>
                    </div>
                  ))
                }
              </div>
            </section>
          )}

          {/* Model Info Section */}
          {modelInfo && (
            <section className="card info-section">
              <h2>ℹ️ ข้อมูลโมเดล</h2>
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">Algorithm</span>
                  <span className="info-value">{modelInfo.algorithm}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Classes</span>
                  <span className="info-value">{modelInfo.classes?.join(', ')}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Vocabulary Size</span>
                  <span className="info-value">{modelInfo.vocabulary_size?.toLocaleString()}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Version</span>
                  <span className="info-value">{modelInfo.version}</span>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Thai News Topic Classifier © 2026 | TF-IDF + Logistic Regression</p>
      </footer>
    </div>
  )
}

export default App
