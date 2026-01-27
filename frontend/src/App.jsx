import React, { useState, useEffect } from 'react';
import {
  Trash2,
  BarChart2,
  Microscope,
  Briefcase,
  Globe,
  Info,
  Cpu,
  Activity,
  Newspaper,
  Shuffle,
  Clock,
  AlertTriangle,
  XCircle
} from 'lucide-react';
import './index.css';

// Import data from JSON files
import SAMPLE_NEWS from './data/sampleNews.json';
import MISCLASSIFIED_EXAMPLES from './data/misclassifiedExamples.json';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

export default function App() {
  const [headline, setHeadline] = useState('');
  const [body, setBody] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [modelInfo, setModelInfo] = useState(null);
  const [showErrorAnalysis, setShowErrorAnalysis] = useState(false);

  useEffect(() => {
    checkHealth();
    fetchModelInfo();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      const data = await res.json();
      setApiStatus(data.status === 'healthy' ? 'online' : 'offline');
    } catch {
      setApiStatus('offline');
    }
  };

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_URL}/model/info`);
      const data = await res.json();
      setModelInfo(data);
    } catch {
      console.log('Could not fetch model info');
    }
  };

  // Try Example - สุ่มข่าวตัวอย่าง
  const handleTryExample = () => {
    const randomIndex = Math.floor(Math.random() * SAMPLE_NEWS.length);
    const sample = SAMPLE_NEWS[randomIndex];
    setHeadline(sample.headline);
    setBody(sample.body);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!headline && !body) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ headline, body })
      });

      const data = await res.json();

      if (res.ok) {
        const iconMap = {
          'Business': <Briefcase className="w-5 h-5" />,
          'SciTech': <Microscope className="w-5 h-5" />,
          'World': <Globe className="w-5 h-5" />
        };
        const colorMap = {
          'Business': { bg: 'bg-blue-500', text: 'text-blue-600' },
          'SciTech': { bg: 'bg-purple-500', text: 'text-purple-600' },
          'World': { bg: 'bg-green-500', text: 'text-green-600' }
        };

        const probabilities = Object.entries(data.probabilities)
          .map(([label, score]) => ({
            label,
            score: score * 100,
            icon: iconMap[label],
            color: colorMap[label]?.bg || 'bg-gray-500',
            text: colorMap[label]?.text || 'text-gray-600'
          }))
          .sort((a, b) => b.score - a.score);

        setResult({
          prediction: probabilities[0],
          probabilities: probabilities,
          latency_ms: data.latency_ms,
          model_version: data.model_version
        });
      } else {
        setError(data.message || data.error || 'เกิดข้อผิดพลาด');
      }
    } catch {
      setError('ไม่สามารถเชื่อมต่อกับ API ได้ กรุณาตรวจสอบว่า Backend กำลังทำงาน');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setHeadline('');
    setBody('');
    setResult(null);
    setError(null);
  };

  // Error Analysis Page
  if (showErrorAnalysis) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-800 font-sans">
        <header className="bg-white shadow-sm sticky top-0 z-10 border-b border-slate-200">
          <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-red-600 rounded-lg text-white">
                <AlertTriangle size={24} />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Error Analysis</h1>
                <p className="text-xs text-slate-500 font-medium">ตัวอย่างที่โมเดลทำนายผิด</p>
              </div>
            </div>
            <button
              onClick={() => setShowErrorAnalysis(false)}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700 font-medium transition-all"
            >
              ← กลับหน้าหลัก
            </button>
          </div>
        </header>

        <main className="max-w-5xl mx-auto px-4 py-8">
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
            <p className="text-amber-800 text-sm">
              <strong>หมายเหตุ:</strong> ตัวอย่างด้านล่างแสดงกรณีที่โมเดลทำนายผิดพลาด พร้อมการวิเคราะห์สาเหตุและข้อเสนอแนะ
            </p>
          </div>

          <div className="space-y-6">
            {MISCLASSIFIED_EXAMPLES.map((example, index) => (
              <div key={index} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                <div className="bg-red-50 px-6 py-3 border-b border-red-100 flex items-center gap-2">
                  <XCircle className="w-4 h-4 text-red-500" />
                  <span className="font-semibold text-red-700">ตัวอย่างที่ {index + 1}</span>
                </div>
                <div className="p-6 space-y-4">
                  <div>
                    <label className="text-xs font-medium text-slate-500 uppercase">Headline</label>
                    <p className="text-slate-800 font-medium">{example.headline}</p>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-500 uppercase">Body</label>
                    <p className="text-slate-600 text-sm">{example.body}</p>
                  </div>
                  <div className="flex gap-4 pt-2">
                    <div className="flex-1 bg-green-50 rounded-lg p-3 border border-green-200">
                      <span className="text-xs text-green-600 font-medium">Actual Label</span>
                      <p className="text-green-800 font-bold text-lg">{example.actual}</p>
                    </div>
                    <div className="flex-1 bg-red-50 rounded-lg p-3 border border-red-200">
                      <span className="text-xs text-red-600 font-medium">Predicted Label</span>
                      <p className="text-red-800 font-bold text-lg">{example.predicted}</p>
                    </div>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                    <span className="text-xs font-medium text-slate-500 uppercase">การวิเคราะห์สาเหตุ</span>
                    <p className="text-slate-700 mt-1">{example.reason}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 bg-indigo-50 border border-indigo-200 rounded-lg p-6">
            <h3 className="font-bold text-indigo-800 mb-3">💡 ข้อเสนอแนะในการปรับปรุง</h3>
            <ul className="space-y-2 text-indigo-700 text-sm">
              <li>• ใช้ Pre-trained Thai Language Model (เช่น WangchanBERTa) เพื่อเข้าใจบริบทดีขึ้น</li>
              <li>• เพิ่มข้อมูล Training ที่มีความหลากหลายมากขึ้น</li>
              <li>• พิจารณาใช้ Multi-label Classification สำหรับข่าวที่มีหลายหมวดหมู่</li>
              <li>• เพิ่ม Feature จาก subtopic เพื่อช่วยแยกแยะ</li>
            </ul>
          </div>
        </main>
      </div>
    );
  }

  // Main Page
  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans selection:bg-indigo-100">
      <header className="bg-white shadow-sm sticky top-0 z-10 border-b border-slate-200">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-600 rounded-lg text-white">
              <Newspaper size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 leading-tight">Thai News Topic Classifier</h1>
              <p className="text-xs text-slate-500 font-medium">ระบบจำแนกหมวดหมู่ข่าวภาษาไทย</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowErrorAnalysis(true)}
              className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-amber-50 text-amber-700 rounded-lg text-xs font-semibold border border-amber-200 hover:bg-amber-100 transition-all"
            >
              <AlertTriangle size={14} />
              Error Analysis
            </button>
            <div className={`hidden sm:flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border ${apiStatus === 'online'
              ? 'bg-green-50 text-green-700 border-green-200'
              : apiStatus === 'offline'
                ? 'bg-red-50 text-red-700 border-red-200'
                : 'bg-gray-50 text-gray-700 border-gray-200'
              }`}>
              <div className={`w-2 h-2 rounded-full ${apiStatus === 'online' ? 'bg-green-500 animate-pulse' :
                apiStatus === 'offline' ? 'bg-red-500' : 'bg-gray-400'
                }`}></div>
              API: {apiStatus === 'online' ? 'ออนไลน์' : apiStatus === 'offline' ? 'ออฟไลน์' : 'กำลังตรวจสอบ...'}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Input Section */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-slate-500" />
                <h2 className="font-semibold text-slate-700">กรอกข้อมูลข่าว</h2>
              </div>
              {/* Try Example Button */}
              <button
                onClick={handleTryExample}
                className="flex items-center gap-2 px-3 py-1.5 bg-indigo-50 text-indigo-600 rounded-lg text-sm font-medium hover:bg-indigo-100 transition-all border border-indigo-200"
              >
                <Shuffle size={16} />
                Try Example
              </button>
            </div>

            <div className="p-6 space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Headline (พาดหัวข่าว)</label>
                <input
                  type="text"
                  value={headline}
                  onChange={(e) => setHeadline(e.target.value)}
                  placeholder="ตัวอย่าง: หุ้นไทยปิดบวกรับข่าวดีเศรษฐกิจฟื้นตัว..."
                  className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all outline-none"
                  disabled={isLoading}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Body (เนื้อหาข่าว)</label>
                <textarea
                  value={body}
                  onChange={(e) => setBody(e.target.value)}
                  placeholder="วางเนื้อหาข่าวที่นี่เพื่อการวิเคราะห์ที่แม่นยำยิ่งขึ้น..."
                  rows={8}
                  className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all outline-none resize-none"
                  disabled={isLoading}
                ></textarea>
              </div>

              <div className="flex gap-3 pt-2">
                <button
                  onClick={handlePredict}
                  disabled={isLoading || (!headline && !body) || apiStatus !== 'online'}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 px-6 rounded-lg font-semibold text-white transition-all shadow-md hover:shadow-lg
                    ${isLoading || (!headline && !body) || apiStatus !== 'online'
                      ? 'bg-slate-400 cursor-not-allowed'
                      : 'bg-indigo-600 hover:bg-indigo-700 active:scale-95'}`}
                >
                  {isLoading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      กำลังวิเคราะห์...
                    </>
                  ) : (
                    <>
                      <Cpu size={20} />
                      Predict
                    </>
                  )}
                </button>

                <button
                  onClick={handleClear}
                  disabled={isLoading}
                  className="flex items-center justify-center gap-2 py-3 px-6 rounded-lg font-semibold text-slate-600 bg-white border border-slate-300 hover:bg-slate-50 hover:text-red-600 transition-all active:scale-95"
                >
                  <Trash2 size={20} />
                  <span className="hidden sm:inline">ล้างข้อมูล</span>
                </button>
              </div>

              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Mobile Error Analysis Link */}
          <button
            onClick={() => setShowErrorAnalysis(true)}
            className="sm:hidden w-full flex items-center justify-center gap-2 p-4 bg-amber-50 text-amber-700 rounded-lg text-sm font-semibold border border-amber-200"
          >
            <AlertTriangle size={16} />
            ดูหน้า Error Analysis
          </button>
        </div>

        {/* Output Section */}
        <div className="lg:col-span-5 space-y-6">
          {/* Main Prediction Card */}
          <div className={`bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden transition-all duration-500 ${result ? 'opacity-100 translate-y-0' : 'opacity-50 translate-y-4'}`}>
            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-slate-500" />
              <h2 className="font-semibold text-slate-700">ผลการทำนาย</h2>
            </div>

            {result ? (
              <div className="p-6">
                <div className="flex flex-col items-center justify-center py-6">
                  <div className={`w-20 h-20 rounded-full flex items-center justify-center mb-4 ${result.prediction.color} bg-opacity-10`}>
                    {React.cloneElement(result.prediction.icon, { className: `w-10 h-10 ${result.prediction.text}` })}
                  </div>
                  <h3 className={`text-3xl font-bold ${result.prediction.text} mb-1`}>{result.prediction.label}</h3>
                  <p className="text-slate-500 font-medium">Confidence: <span className="text-slate-900">{result.prediction.score.toFixed(1)}%</span></p>
                </div>

                {/* System Info: Latency & Model Version */}
                <div className="flex gap-3 mb-4">
                  <div className="flex-1 bg-slate-50 rounded-lg p-3 text-center border border-slate-200">
                    <div className="flex items-center justify-center gap-1 text-slate-500 text-xs mb-1">
                      <Clock size={12} />
                      Latency
                    </div>
                    <p className="font-bold text-slate-800">{result.latency_ms} ms</p>
                  </div>
                  <div className="flex-1 bg-slate-50 rounded-lg p-3 text-center border border-slate-200">
                    <div className="flex items-center justify-center gap-1 text-slate-500 text-xs mb-1">
                      <Info size={12} />
                      Model Version
                    </div>
                    <p className="font-bold text-slate-800">{result.model_version}</p>
                  </div>
                </div>

                <div className="space-y-4 border-t border-slate-100 pt-6">
                  <h4 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">ความน่าจะเป็นแต่ละหมวด</h4>
                  {result.probabilities.map((item, index) => (
                    <div key={index} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="flex items-center gap-2 font-medium text-slate-700">
                          {item.icon}
                          {item.label}
                        </span>
                        <span className="font-bold text-slate-900">{item.score.toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${item.color} transition-all duration-1000 ease-out`}
                          style={{ width: `${item.score}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-12 text-center text-slate-400">
                <div className="mx-auto w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                  <BarChart2 className="w-8 h-8 text-slate-300" />
                </div>
                <p>รอการประมวลผลข้อมูล...</p>
              </div>
            )}
          </div>

          {/* Model Info Card */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <Info className="w-4 h-4 text-indigo-500" />
              ข้อมูลโมเดล
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Algorithm</span>
                <span className="font-medium text-slate-900">{modelInfo?.algorithm || 'TF-IDF + Logistic Regression'}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Classes</span>
                <span className="font-medium text-slate-900 text-right">{modelInfo?.classes?.join(', ') || 'Business, SciTech, World'}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Vocabulary Size</span>
                <span className="font-medium text-slate-900">{modelInfo?.vocabulary_size?.toLocaleString() || '-'}</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-slate-500">Version</span>
                <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs font-bold">{modelInfo?.version || '1.0.0'}</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="max-w-5xl mx-auto px-4 py-8 text-center text-slate-400 text-sm">
        <p>Thai News Topic Classifier © 2026 | TF-IDF + Logistic Regression</p>
      </footer>
    </div>
  );
}
