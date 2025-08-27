import {
  useEffect,
  useRef
} from 'react'

const AudioPlayer = ({ audioUrl, mimeType }) => {
  const audioPlayer = useRef(null)
  const audioSource = useRef(null);

  useEffect(() => {
    if (audioPlayer.current && audioSource.current) {
      audioSource.current.src = audioUrl;
      audioPlayer.current.load();
    }
  }, [audioUrl])
  
  return (
    <div className="flex relative z-10 my-4 w-full">
      <audio
        ref={audioPlayer}
        controls
        className="w-full h-14 rounded-lg bg-white 
              shadow-xl shadow-black/5 ring-1 ring-slate-700/10"
      >
        {/* 浏览器兼容声音的类型不一样 多个source */}
        <source ref={audioSource} src={audioUrl} type={mimeType} />
        Your browser does not support the audio element.
      </audio>
    </div>
  )
}

export default AudioPlayer