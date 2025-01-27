/**
 *Submitted for verification at Etherscan.io on 2016-03-26
*/

contract Rating {
        function setRating(bytes32 _key, uint256 _value) {
            ratings[_key] = _value;
        }
        mapping (bytes32 => uint256) public ratings;
    }