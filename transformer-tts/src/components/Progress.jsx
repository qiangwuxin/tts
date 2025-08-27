//进度条组件

const Progress = ({ text, percentage = 0 }) => {
  return (
    <div className="relative text-black bg-white rounded-lg text-left overflow-hidden mb-2">
      <div
        className='px-2 h-full bg-blue-500 whitespace-nowrap transition-all duration-300'
        style={{ width: `${percentage}%` }}>
        {text} - {percentage.toFixed(2)}%
      </div>
    </div>
  )
}
export default Progress