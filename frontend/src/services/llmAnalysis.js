/**
 * LLM Error Analysis Service
 * เรียก Azure OpenAI Chat Completions API เพื่อวิเคราะห์ว่าทำไมโมเดลถึงทำนายแบบนี้
 */

const AZURE_ENDPOINT = import.meta.env.VITE_AZURE_OPENAI_ENDPOINT;
const AZURE_API_KEY = import.meta.env.VITE_AZURE_OPENAI_API_KEY;
const AZURE_API_VERSION = import.meta.env.VITE_AZURE_OPENAI_API_VERSION || '2024-04-01-preview';
const AZURE_DEPLOYMENT = import.meta.env.VITE_AZURE_OPENAI_DEPLOYMENT || 'gpt-5.2-chat';
const DISCORD_WEBHOOK_URL = import.meta.env.VITE_DISCORD_WEBHOOK_URL;

// Rate Limiting Config
const RATE_LIMIT_MAX_REQUESTS = 1;  // จำนวนครั้งสูงสุด
const RATE_LIMIT_WINDOW_MS = 60000; // 1 นาที (60000 ms)
const STORAGE_KEY = 'llm_analysis_rate_limit';
const DEVICE_ID_KEY = 'llm_device_id';

/**
 * สร้างหรือดึง Device ID
 */
function getDeviceId() {
    let deviceId = localStorage.getItem(DEVICE_ID_KEY);
    if (!deviceId) {
        deviceId = 'device_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
        localStorage.setItem(DEVICE_ID_KEY, deviceId);
    }
    return deviceId;
}

/**
 * ดึง IP Address ของผู้ใช้
 */
async function getUserIP() {
    try {
        const res = await fetch('https://api.ipify.org?format=json');
        const data = await res.json();
        return data.ip;
    } catch {
        return 'Unknown';
    }
}

/**
 * ส่ง Log ไปยัง Discord Webhook
 * @param {Object} params - ข้อมูลสำหรับ log
 */
async function sendDiscordLog({ headline, body, prediction, trueLabel, analysis }) {
    if (!DISCORD_WEBHOOK_URL) {
        console.warn('Discord webhook URL not configured');
        return;
    }

    try {
        const ip = await getUserIP();
        const time = new Date().toLocaleString('th-TH', { timeZone: 'Asia/Bangkok' });
        const deviceId = getDeviceId();

        // Helper function - Discord requires non-empty field values (min 1 char)
        const safeValue = (str, maxLen = 1024) => {
            if (!str || str.trim() === '') return 'N/A';
            return str.length > maxLen ? str.substring(0, maxLen - 3) + '...' : str;
        };

        // สร้าง embed สำหรับ Discord
        const embed = {
            title: '🔍 AI Error Analysis Log',
            color: 0x7C3AED, // สีม่วง
            fields: [
                {
                    name: '🖥️ Device Info',
                    value: `**IP:** ${ip}\n**Device ID:** ${deviceId}\n**Time:** ${time}`,
                    inline: false
                },
                {
                    name: '📰 Headline',
                    value: safeValue(headline, 1024),
                    inline: false
                },
                {
                    name: '📝 Body (Preview)',
                    value: safeValue(body, 500),
                    inline: false
                },
                {
                    name: '🤖 Model Prediction',
                    value: `**Topic:** ${prediction?.label || 'N/A'}\n**Confidence:** ${prediction?.confidence ? (prediction.confidence * 100).toFixed(1) + '%' : 'N/A'}`,
                    inline: true
                },
                {
                    name: '👤 User Selected',
                    value: trueLabel || 'ไม่ได้ระบุ',
                    inline: true
                },
                {
                    name: '✨ AI Analysis',
                    value: safeValue(analysis, 1024),
                    inline: false
                }
            ],
            timestamp: new Date().toISOString(),
            footer: {
                text: 'Thai News Topic Classifier'
            }
        };

        await fetch(DISCORD_WEBHOOK_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ embeds: [embed] })
        });

        console.log('Discord log sent successfully');
    } catch (error) {
        console.error('Failed to send Discord log:', error);
    }
}

/**
 * ตรวจสอบ Rate Limit
 * @returns {{allowed: boolean, remainingRequests: number, resetTime: number}}
 */
function checkRateLimit() {
    const now = Date.now();
    const deviceId = getDeviceId();

    // ดึงข้อมูล rate limit จาก localStorage
    let rateLimitData = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');

    // ดึงข้อมูลสำหรับ device นี้
    let deviceData = rateLimitData[deviceId] || { requests: [] };

    // ลบ requests ที่เก่าเกิน window
    deviceData.requests = deviceData.requests.filter(
        timestamp => (now - timestamp) < RATE_LIMIT_WINDOW_MS
    );

    const remainingRequests = RATE_LIMIT_MAX_REQUESTS - deviceData.requests.length;
    const oldestRequest = deviceData.requests[0] || now;
    const resetTime = Math.max(0, RATE_LIMIT_WINDOW_MS - (now - oldestRequest));

    return {
        allowed: remainingRequests > 0,
        remainingRequests: Math.max(0, remainingRequests),
        resetTime: Math.ceil(resetTime / 1000), // seconds
        deviceId
    };
}

/**
 * บันทึก request ใหม่
 */
function recordRequest() {
    const now = Date.now();
    const deviceId = getDeviceId();

    let rateLimitData = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    let deviceData = rateLimitData[deviceId] || { requests: [] };

    // ลบ requests เก่า
    deviceData.requests = deviceData.requests.filter(
        timestamp => (now - timestamp) < RATE_LIMIT_WINDOW_MS
    );

    // เพิ่ม request ใหม่
    deviceData.requests.push(now);

    // บันทึกกลับ
    rateLimitData[deviceId] = deviceData;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(rateLimitData));
}

/**
 * วิเคราะห์ผลการทำนายด้วย LLM
 * @param {Object} params - ข้อมูลสำหรับวิเคราะห์
 * @param {string} params.headline - พาดหัวข่าว
 * @param {string} params.body - เนื้อหาข่าว
 * @param {Object} params.prediction - ผลการทำนายจากโมเดล
 * @param {string} [params.trueLabel] - Label ที่ถูกต้อง (optional)
 * @returns {Promise<{analysis: string, error?: string, rateLimit?: object}>}
 */
export async function analyzePrediction({ headline, body, prediction, trueLabel }) {
    // ตรวจสอบ Rate Limit ก่อน
    const rateLimit = checkRateLimit();
    if (!rateLimit.allowed) {
        return {
            error: `⚠️ เกินจำนวนครั้งที่อนุญาต (${RATE_LIMIT_MAX_REQUESTS} ครั้ง/นาที)\nกรุณารอ ${rateLimit.resetTime} วินาที`,
            analysis: null,
            rateLimit
        };
    }

    if (!AZURE_API_KEY || !AZURE_ENDPOINT) {
        return {
            error: 'Azure OpenAI credentials not configured',
            analysis: null
        };
    }

    // สร้าง Prompt สำหรับ LLM
    const prompt = buildPrompt({ headline, body, prediction, trueLabel });

    // Azure OpenAI Chat Completions API URL
    const url = `${AZURE_ENDPOINT}/openai/deployments/${AZURE_DEPLOYMENT}/chat/completions?api-version=${AZURE_API_VERSION}`;

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'api-key': AZURE_API_KEY
            },
            body: JSON.stringify({
                messages: [
                    {
                        role: 'system',
                        content: 'คุณคือผู้ช่วยวิเคราะห์ผลการทำนายของโมเดลจำแนกหมวดหมู่ข่าวภาษาไทย ตอบเป็นภาษาไทย กระชับ ใช้ bullet points'
                    },
                    {
                        role: 'user',
                        content: prompt
                    }
                ],
                max_completion_tokens: 1024
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('Azure OpenAI Error:', errorData);
            throw new Error(errorData.error?.message || `HTTP ${response.status}`);
        }

        const data = await response.json();

        // Extract text from Chat Completions response
        const analysis = data.choices?.[0]?.message?.content || 'ไม่สามารถวิเคราะห์ได้';

        // บันทึก request สำเร็จ
        recordRequest();

        // ส่ง Log ไปยัง Discord (fire and forget)
        sendDiscordLog({ headline, body, prediction, trueLabel, analysis });

        // ดึงข้อมูล rate limit ล่าสุด
        const updatedRateLimit = checkRateLimit();

        return { analysis, error: null, rateLimit: updatedRateLimit };
    } catch (error) {
        console.error('LLM Error Analysis failed:', error);
        return {
            error: error.message || 'ไม่สามารถเรียก LLM ได้',
            analysis: null
        };
    }
}

/**
 * สร้าง Prompt สำหรับ LLM
 */
function buildPrompt({ headline, body, prediction, trueLabel }) {
    const probsText = Object.entries(prediction.probabilities || {})
        .map(([label, score]) => `  ${label}: ${(score * 100).toFixed(1)}%`)
        .join('\n');

    let prompt = `ข้อมูลข่าว:
- Headline: ${headline}
- Body: ${body.substring(0, 500)}${body.length > 500 ? '...' : ''}

ผลทำนาย:
- Predicted label: ${prediction.label}
- Confidence: ${(prediction.confidence * 100).toFixed(1)}%
- Probabilities:
${probsText}
- Latency: ${prediction.latency_ms} ms
- Model: ${prediction.model_type || 'ONNX Runtime'} v${prediction.model_version}`;

    if (trueLabel && trueLabel !== prediction.label) {
        prompt += `

Label ที่ถูกต้อง: ${trueLabel} (โมเดลทายผิด)

งานของคุณ:
1) สรุปใจความข่าวแบบสั้น ๆ (1-2 ประโยค)
2) วิเคราะห์ว่าเหตุใดโมเดลถึงทำนายเป็น "${prediction.label}"
3) อธิบายสาเหตุที่โมเดลทายผิด (ทำไมไม่ใช่ "${trueLabel}")
4) แนะนำวิธีปรับปรุงโมเดลให้แม่นยำขึ้น`;
    } else {
        prompt += `

งานของคุณ:
1) สรุปใจความข่าวแบบสั้น ๆ (1-2 ประโยค)
2) วิเคราะห์ว่าเหตุใดโมเดลถึงทำนายเป็น "${prediction.label}"
3) อธิบายคำศัพท์/บริบทสำคัญที่ทำให้โมเดลตัดสินใจ
4) ประเมินว่าการทำนายนี้น่าเชื่อถือหรือไม่`;
    }

    return prompt;
}

export { checkRateLimit };
export default { analyzePrediction, checkRateLimit };
